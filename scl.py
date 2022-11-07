
import torch
from args import args

def train_epoch_scl(encoder, projector, train_loader, train_transform, criterion, optimizer, scheduler):

    epoch_loss = 0.0

    for data, target, _ in train_loader:

        with torch.no_grad():
            data_t1 = train_transform(data)
            data_t2 = train_transform(data)

        feat1, feat2 = encoder(data_t1), encoder(data_t2)
        proj1, proj2 = projector(feat1), projector(feat2)

        loss = criterion(proj1, proj2, target)

        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = epoch_loss / len(train_loader)
    scheduler.step()
    
    return epoch_loss

def train_scl(encoder, projector, train_loader, train_transform, criterion, optimizer, scheduler, epochs):

    best_loss = None
    train_losses = []

    encoder.train()
    projector.train()

    for i in range(1, epochs+1):

        print(f"Epoch {i}")
        train_loss = train_epoch_scl(encoder, projector, train_loader, train_transform, criterion, optimizer, scheduler)
        print(f"Current Train Loss : {format(train_loss, '.4f')}")
        train_losses.append(train_loss)  

        if best_loss is None:
            best_loss = train_loss
        if best_loss > train_loss:
            best_loss = train_loss
            
    final_state = {"encoder": encoder.state_dict()}
    print(f"Last Loss : {format(train_loss, '.4f')}\tBest Loss : {format(best_loss, '.4f')}")

    return train_losses, encoder, final_state

def linear_train_epoch(encoder, classifier, train_loader, val_transform, criterion, optimizer):

    epoch_loss = 0.0
    
    TP = [0, 0, 0 ,0]
    GT = [0, 0, 0, 0]

    classifier.train()

    for data, target, _ in train_loader:
        data, target = data.to(args.device), target.to(args.device)  

        with torch.no_grad():
            features = encoder(val_transform(data))
        
        optimizer.zero_grad()

        output = classifier(features)
        loss = criterion(output, target)
            
        epoch_loss += loss.item()

        _, labels_predicted = torch.max(output, dim=1)

        for idx in range(len(TP)):
            TP[idx] += torch.logical_and((labels_predicted==idx),(target==idx)).sum().item()
            GT[idx] += (target==idx).sum().item()
    
        loss.backward()
        optimizer.step()

    epoch_loss = epoch_loss / len(train_loader)
    se = sum(TP[1:])/sum(GT[1:])
    sp = TP[0]/GT[0]
    icbhi_score = (se+sp)/2
    acc = sum(TP)/sum(GT)

    return epoch_loss, se, sp, icbhi_score, acc

def linear_eval_epoch(encoder, classifier, val_loader, val_transform, criterion):

    epoch_loss = 0.0

    TP = [0, 0, 0 ,0]
    GT = [0, 0, 0, 0]

    classifier.eval()
    encoder.eval()

    with torch.no_grad():

        for data, target, _ in val_loader:
            data, target = data.to(args.device), target.to(args.device)
            
            output = classifier(encoder(val_transform(data)))
            loss = criterion(output, target)
            epoch_loss += loss.item()

            _, labels_predicted = torch.max(output, dim=1)

            for idx in range(len(TP)):
                TP[idx] += torch.logical_and((labels_predicted==idx),(target==idx)).sum().item()
                GT[idx] += (target==idx).sum().item()

    epoch_loss = epoch_loss / len(val_loader)
    se = sum(TP[1:])/sum(GT[1:])
    sp = TP[0]/GT[0]
    icbhi_score = (se+sp)/2
    acc = sum(TP)/sum(GT)

    return epoch_loss, se, sp, icbhi_score, acc

def linear_scl(encoder, checkpoint, classifier, train_loader, val_loader, val_transform, criterion, optimizer, epochs):

    train_losses = []; val_losses = []; train_se_scores = []; train_sp_scores = []; train_icbhi_scores = []; train_acc_scores = []; val_se_scores = []; val_sp_scores = []; val_icbhi_scores = []; val_acc_scores = []

    best_val_acc = 0
    best_icbhi_score = 0
    best_se = 0
    best_sp = 0
    best_epoch_acc = 0
    best_epoch_icbhi = 0

    state_dict = checkpoint["encoder"]
    encoder.load_state_dict(state_dict)

    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()

    for i in range(1, epochs+1):

        print(f"Epoch {i}")

        train_loss, train_se, train_sp, train_icbhi_score, train_acc = linear_train_epoch(encoder, classifier, train_loader, val_transform, criterion, optimizer)
        train_losses.append(train_loss); train_se_scores.append(train_se); train_sp_scores.append(train_sp); train_icbhi_scores.append(train_icbhi_score); train_acc_scores.append(train_acc)
        print(f"Train loss : {format(train_loss, '.4f')}\tTrain SE : {format(train_se, '.4f')}\tTrain SP : {format(train_sp, '.4f')}\tTrain Score : {format(train_icbhi_score, '.4f')}\tTrain Acc : {format(train_acc, '.4f')}")

        val_loss, val_se, val_sp, val_icbhi_score, val_acc = linear_eval_epoch(encoder, classifier, val_loader, val_transform, criterion)
        val_losses.append(val_loss); val_se_scores.append(val_se); val_sp_scores.append(val_sp); val_icbhi_scores.append(val_icbhi_score); val_acc_scores.append(val_acc)
        print(f"Val loss : {format(val_loss, '.4f')}\tVal SE : {format(val_se, '.4f')}\tVal SP : {format(val_sp, '.4f')}\tVal Score : {format(val_icbhi_score, '.4f')}\tVal Acc : {format(val_acc, '.4f')}")

        if best_val_acc == 0:
            best_val_acc = val_acc

        if i == 1:
            best_icbhi_score = val_icbhi_score
            best_se = val_se
            best_sp = val_sp

        if best_icbhi_score < val_icbhi_score:
            best_epoch_icbhi = i
            best_icbhi_score = val_icbhi_score
            best_se = val_se
            best_sp = val_sp

        if best_val_acc < val_acc:
            best_epoch_acc = i
            best_val_acc = val_acc
        
    print(f"best score:{format(best_icbhi_score, '.4f')} (se:{format(best_se, '.4f')} sp:{format(best_sp, '.4f')}) at epoch {format(best_epoch_icbhi, '.4f')}")

    return train_losses, val_losses, train_se_scores, train_sp_scores, train_icbhi_scores, train_acc_scores, val_se_scores, val_sp_scores, val_icbhi_scores, val_acc_scores