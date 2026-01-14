import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from tqdm import tqdm

# Importamos nuestros módulos locales
from model import UNetMultibanda
from losses import DiceLoss, BCEDiceLoss
from utils import calculate_iou, calculate_accuracy

class Entrenador:
    def __init__(self, modelo, train_loader, val_loader, config, maestro=None):
        """
        config: Diccionario con llaves:
            'optimizador': str ('adam', 'sgd', 'rmsprop')
            'lr': float
            'loss': str ('bce', 'dice', 'bce_dice')
            'device': str
            'epochs': int
            'distillation': bool
            'alpha_distill': float (peso de la pérdida del maestro)
            'temperature': float
            'name': str (nombre del experimento)
        maestro: Modelo pre-entrenado (opcional) para Knowledge Distillation
        """
        self.model = modelo
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(config['device'])
        self.model.to(self.device)
        
        # Configuración de Teacher (Distillation)
        self.teacher = maestro
        if self.config.get('distillation') and self.teacher:
            self.teacher.to(self.device)
            self.teacher.eval() # Congelado
            print("Knowledge Distillation Activado")

        # Configurar Pérdida
        self.criterion = self._get_loss_function(config['loss'])
        
        # Configurar Optimizador
        self.optimizer = self._get_optimizer(config['optimizador'], config['lr'])
        
        # TensorBoard
        time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_name = f"{config['name']}_{config['optimizador']}_{config['loss']}_{time_str}"
        self.writer = SummaryWriter(log_dir=f"runs/{exp_name}")
        self.global_step = 0

    def _get_loss_function(self, name):
        if name == 'bce': return torch.nn.BCELoss()
        if name == 'dice': return DiceLoss()
        if name == 'bce_dice': return BCEDiceLoss()
        raise ValueError(f"Loss {name} no soportada")

    def _get_optimizer(self, name, lr):
        if name == 'adam': return optim.Adam(self.model.parameters(), lr=lr)
        if name == 'sgd': return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        if name == 'rmsprop': return optim.RMSprop(self.model.parameters(), lr=lr)
        raise ValueError(f"Optimizador {name} no soportado")

    def _distillation_loss(self, student_pred, teacher_pred, target):
        """
        Calcula la pérdida combinada:
        L = (1 - alpha) * L_hard(student, target) + alpha * L_soft(student, teacher)
        """
        alpha = self.config.get('alpha_distill', 0.5)
        
        # Pérdida dura (contra la máscara real)
        hard_loss = self.criterion(student_pred, target)
        
        # Pérdida suave (Student debe imitar al Teacher)
        # Usamos MSE para comparar probabilidades o KLDiv
        soft_loss = torch.nn.MSELoss()(student_pred, teacher_pred)
        
        return (1 - alpha) * hard_loss + alpha * soft_loss

    def train_epoch(self, epoch_index):
        self.model.train()
        running_loss = 0.0
        
        loop = tqdm(self.train_loader, desc=f"Epoch {epoch_index}")
        
        for batch in loop:
            # Nota: Adaptado a la estructura de tu dataload.ipynb
            # batch es un dict con 'input', 'mask', 'target'
            images = batch['input'].to(self.device)
            # Para Modelo A, el target es la máscara de nubes 'mask'
            targets = batch['mask'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images) # Salida Sigmoid (0-1)
            
            loss = 0
            
            # Lógica de Distillation
            if self.config.get('distillation') and self.teacher:
                with torch.no_grad():
                    teacher_out = self.teacher(images)
                loss = self._distillation_loss(outputs, teacher_out, targets)
            else:
                loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
            # Logging paso a paso
            self.writer.add_scalar('Train/Loss_Step', loss.item(), self.global_step)
            self.global_step += 1
            
        avg_loss = running_loss / len(self.train_loader)
        self.writer.add_scalar('Train/Loss_Epoch', avg_loss, epoch_index)
        return avg_loss

    def validate(self, epoch_index):
        self.model.eval()
        val_iou = 0.0
        val_acc = 0.0
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['input'].to(self.device)
                targets = batch['mask'].to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                val_loss += loss.item()
                val_iou += calculate_iou(outputs, targets)
                val_acc += calculate_accuracy(outputs, targets)
        
        n = len(self.val_loader)
        self.writer.add_scalar('Val/IoU', val_iou / n, epoch_index)
        self.writer.add_scalar('Val/Accuracy', val_acc / n, epoch_index)
        self.writer.add_scalar('Val/Loss', val_loss / n, epoch_index)
        
        # Guardar imágenes de ejemplo en TensorBoard (RGB channels only)
        # Sentinel-2 Bands: [B04, B03, B02, B08, B11, B12] -> Indices 0,1,2 son RGB
        rgb_img = images[0, :3, :, :] # Tomar solo la primera imagen del batch y canales RGB
        self.writer.add_image('Visual/Input_RGB', rgb_img, epoch_index)
        self.writer.add_image('Visual/Ground_Truth', targets[0], epoch_index)
        self.writer.add_image('Visual/Prediction', outputs[0], epoch_index)
        
        # Histograma de pesos (Opcional, consume espacio)
        for name, param in self.model.named_parameters():
             if 'weight' in name:
                 self.writer.add_histogram(f'Weights/{name}', param, epoch_index)

        print(f"Val Loss: {val_loss/n:.4f} | IoU: {val_iou/n:.4f}")

    def run(self):
        print(f"Iniciando experimento: {self.config['name']}")
        for epoch in range(self.config['epochs']):
            self.train_epoch(epoch)
            self.validate(epoch)
        
        self.writer.close()
        print("Entrenamiento finalizado.")