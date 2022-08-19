from tqdm import tqdm
import torch

class Trainer:
    
    def __init__(self, model, train_loader, optimizer, loss_fn, early_stopping, evaluator, lr_scheduler, 
                 util, device, verbose=False):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.early_stopping = early_stopping
        self.evaluator = evaluator
        self.lr_scheduler = lr_scheduler
        self.util = util
        self.device = device
        self.verbose = verbose
        
    def fit(self, epochs, val_check_interval=None):
        train_losses, val_losses = [], []
        for epoch in range(1, epochs + 1):
            train_loss = self.__train(epoch, val_check_interval)
            val_loss = self.evaluator.evaluate()
            
            if self.verbose:
                print(f'Epoch: {epoch}/{epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}')
                
            self.util.save_losses(epoch, train_loss, val_loss)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if self.lr_scheduler:
                self.lr_scheduler.step(val_loss)
            
            self.early_stopping(val_loss, self.model, self.optimizer, epoch)

            if (torch.cuda.is_available()):
                torch.cuda.empty_cache()
            
            if self.early_stopping.isToStop:
                if (self.verbose):
                    print("=> Stopped")
                break

        return train_losses, val_losses
        
    def __train(self, epoch, val_check_interval=None):
        self.model.train()
        epoch_loss, cumulative_loss = 0.0, 0.0
        
        total_iter = len(self.train_loader)
        
        pbar = tqdm(enumerate(self.train_loader, 1), total=total_iter)
        
        for batch_i, (inputs, target) in pbar:
            inputs, target = inputs.to(self.device), target.to(self.device)
            output = self.model(inputs)

            loss = self.loss_fn(output, target)
            pbar.set_postfix({'loss': loss.item()})
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            cumulative_loss += loss.item()
            
            if val_check_interval:
                if (batch_i % val_check_interval) == 0:
                    cumulative_loss = cumulative_loss / val_check_interval
                    val_loss = self.evaluator.evaluate()
                    
                    self.util.save_losses(epoch - 1, cumulative_loss, val_loss)
                    
                    if self.verbose:
                        print(f'train_loss: {cumulative_loss:.4f} - val_loss: {val_loss:.4f} - loss: {loss:.4f}')
                    
                    self.early_stopping(val_loss, self.model, self.optimizer, epoch)
                    self.model.train()
                    
                    if self.early_stopping.isToStop:
                        if (self.verbose):
                            print("=> Stopped")
                        break
                    cumulative_loss = 0.0

        return epoch_loss/total_iter
    

class EarlyStopping:
    
    def __init__(self, filename, patience=15, no_stop=False, verbose=False):
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
        self.isToStop = False
        self.enable_stop = not no_stop
        self.filename = filename
        self.verbose = verbose
          
    def __call__(self, val_loss, model, optimizer, epoch):
        is_best = bool(val_loss < self.best_loss)
        if (is_best):
            self.best_loss = val_loss
            self.__save_checkpoint(self.best_loss, model, optimizer, epoch)
            self.counter = 0
        elif (self.enable_stop):
            self.counter += 1
            if (self.verbose):
                print(f'=> Early stopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.isToStop = True
    
    def __save_checkpoint(self, loss, model, optimizer, epoch):
        state = {'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'epoch': epoch,
                 'loss': loss}
        torch.save(state, self.filename)
        if (self.verbose):
            print ('=> Saving a new best') 