# train_b.py
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import itertools

# Importamos módulos locales
from models_b import obtener_modelo_b
from discriminator import PatchGANDiscriminator
from losses_b import GANLoss, PerceptualLoss
# Suponemos que tienes un archivo metrics.py con calculate_psnr
# from metrics import calculate_psnr, calculate_ssim

class EntrenamientoB:
    def __init__(self, train_loader, val_loader, config, modelo_a=None):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Inicializar Generador (Modelo B)
        self.netG = obtener_modelo_b(config['modelo_tipo'], self.device)
        
        # 2. Inicializar Discriminador
        self.netD = PatchGANDiscriminator(in_channels=13).to(self.device)
        
        # 3. Integración Modelo A (Si existe, se usa para inferir máscaras en vuelo)
        self.netA = modelo_a
        if self.netA:
            self.netA.to(self.device).eval()
            for p in self.netA.parameters(): p.requires_grad = False
            print("✅ Modelo A integrado para generación de máscaras.")
        
        # 4. Optimizadores
        lr = config['lr']
        self.optG = optim.Adam(self.netG.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optD = optim.Adam(self.netD.parameters(), lr=lr, betas=(0.5, 0.999))
        
        # 5. Funciones de Pérdida
        self.criterionGAN = GANLoss(config['loss_adv']).to(self.device)
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionPerceptual = PerceptualLoss(self.device)
        
        # Pesos de pérdidas
        self.lambda_L1 = config.get('lambda_l1', 100)
        self.lambda_perceptual = config.get('lambda_perceptual', 10)
        
        # Logger
        self.writer = SummaryWriter(f"runs/ModeloB_{config['modelo_tipo']}_{config['loss_adv']}")
        self.loader = train_loader
        self.val_loader = val_loader
        
    def _get_mask(self, cloudy_img, gt_mask):
        """
        Si hay Modelo A, úsalo. Si no, usa la máscara del dataset (entrenamiento supervisado ideal).
        Para hacer robusto al modelo B, a veces es bueno usar la máscara real.
        """
        if self.netA and self.config.get('use_model_a_mask', False):
            with torch.no_grad():
                pred_mask = self.netA(cloudy_img)
                # Binarizar máscara
                return (pred_mask > 0.5).float()
        return gt_mask

    def run(self):
        steps = 0
        for epoch in range(self.config['epochs']):
            loop = tqdm(self.loader, desc=f"Epoch {epoch}/{self.config['epochs']}")
            
            for batch in loop:
                # Datos: input (nublado), mask (real), target (limpio)
                real_cloudy = batch['input'].to(self.device) # 6 canales
                real_clean = batch['target'].to(self.device) # 6 canales
                
                # Obtener mascara (ground truth o inferida por A)
                mask = batch['mask'].to(self.device) # 1 canal
                
                # Input del Generador: Concatenar Nublada + Mascara
                # Asegurar dimensiones: Mask (B, 1, H, W)
                if mask.dim() == 3: mask = mask.unsqueeze(1)
                    
                g_input = torch.cat([real_cloudy, mask], dim=1) # 7 canales

                # ---------------------
                #  Entrenar Discriminador
                # ---------------------
                self.optD.zero_grad()
                
                # Generar imagen falsa
                fake_clean = self.netG(g_input)
                
                # Discriminador ve: (Input Condicional + Imagen)
                # Real: (Nublada+Mascara + LimpiaReal)
                real_pair = torch.cat([g_input, real_clean], dim=1)
                pred_real = self.netD(real_pair)
                loss_D_real = self.criterionGAN(pred_real, True)
                
                # Fake: (Nublada+Mascara + LimpiaFalsa)
                fake_pair = torch.cat([g_input, fake_clean.detach()], dim=1)
                pred_fake = self.netD(fake_pair)
                loss_D_fake = self.criterionGAN(pred_fake, False)
                
                loss_D = (loss_D_real + loss_D_fake) * 0.5
                loss_D.backward()
                self.optD.step()

                # -----------------
                #  Entrenar Generador
                # -----------------
                self.optG.zero_grad()
                
                # Adversarial loss (D debe creer que es real)
                fake_pair_g = torch.cat([g_input, fake_clean], dim=1)
                pred_fake_g = self.netD(fake_pair_g)
                loss_G_GAN = self.criterionGAN(pred_fake_g, True)
                
                # Reconstruction Loss (L1)
                # Opcional: Masked L1 (dar mas peso a la zona reconstruida)
                if self.config.get('masked_loss', False):
                    # Perdida en nubes + 0.1 * perdida en zona limpia
                    diff = torch.abs(fake_clean - real_clean)
                    loss_G_L1 = (diff * mask).mean() + 0.1 * (diff * (1-mask)).mean()
                else:
                    loss_G_L1 = self.criterionL1(fake_clean, real_clean)
                
                # Perceptual Loss
                loss_G_Perceptual = self.criterionPerceptual(fake_clean, real_clean)
                
                # Total Loss
                loss_G = loss_G_GAN + (loss_G_L1 * self.lambda_L1) + (loss_G_Perceptual * self.lambda_perceptual)
                
                loss_G.backward()
                self.optG.step()
                
                # Logging
                if steps % 100 == 0:
                    self.writer.add_scalar('Loss/D', loss_D.item(), steps)
                    self.writer.add_scalar('Loss/G_Total', loss_G.item(), steps)
                    self.writer.add_scalar('Loss/G_L1', loss_G_L1.item(), steps)
                    
                steps += 1
            
            # Validación visual al final de la época
            self._validate_visual(g_input, real_clean, fake_clean, epoch)
            
    def _validate_visual(self, g_input, real, fake, epoch):
        # Tomar canales RGB (0,1,2) para visualizar
        # g_input tiene 7 canales: 0-5 bandas, 6 máscara
        input_rgb = g_input[0, :3, :, :].cpu()
        mask_vis = g_input[0, 6, :, :].cpu().unsqueeze(0)
        target_rgb = real[0, :3, :, :].cpu()
        pred_rgb = fake[0, :3, :, :].cpu()
        
        # Desnormalizar si usaste Tanh (-1 a 1) -> (0 a 1)
        input_rgb = (input_rgb + 1) / 2
        target_rgb = (target_rgb + 1) / 2
        pred_rgb = (pred_rgb + 1) / 2
        
        self.writer.add_image('Val/1_Input_Cloudy', input_rgb, epoch)
        self.writer.add_image('Val/2_Mask', mask_vis, epoch)
        self.writer.add_image('Val/3_Target_Clean', target_rgb, epoch)
        self.writer.add_image('Val/4_Reconstructed', pred_rgb, epoch)

# Configuración del Experimento
config_pix2pix = {
    'modelo_tipo': 'pix2pix', # o 'transformer'
    'epochs': 20,
    'lr': 0.0002,
    'loss_adv': 'lsgan',      # 'bce' o 'lsgan'
    'lambda_l1': 100,         # Peso reconstrucción pixel
    'lambda_perceptual': 10,  # Peso estilo/textura
    'masked_loss': True,      # Enfocar perdida en nubes
    'use_model_a_mask': False # False = Usar ground truth para train estable
}

# Suponiendo que tienes train_loader definido en tu notebook anterior
# train_loader debe devolver {'input': img_nublada, 'target': img_limpia, 'mask': mascara}

exp = EntrenamientoB(
    train_loader=train_loader, 
    val_loader=None, # Opcional
    config=config_pix2pix,
    modelo_a=None # Aquí podrías pasar tu modelo_a entrenado
)

print("Iniciando entrenamiento del Modelo B...")
exp.run()