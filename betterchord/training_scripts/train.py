import os
import torch
import torch.nn as nn
import torch.optim as optim

from cnn_model import ChordCNN
from database import get_dataloader

import random
import numpy as np

# Device - use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Training on: {device}")

# Metaparameters
data_root = os.path.join(os.path.dirname(__file__), '..', 'data')
epochs = 150
batch_size = 32
learning_rate = 0.0005
path = os.path.join(os.path.dirname(__file__), 'chord_cnn.pth')
THRESHOLD = 0.48         # must match THRESHOLD in music_theory.py and test_chords.py
ROOT_LOSS_WEIGHT = 0.5   # how much root loss contributes vs note loss
BASS_LOSS_WEIGHT = 0.5   # how much bass loss contributes vs note loss


# Get dataloaders
train_loader, test_loader, note_classes = get_dataloader(data_root, batch_size=batch_size)

# Model:
    # For note recogniztion, 12 vector array for each note in chromatic scale, BCEWithLogitsLoss for multi label
    # For root recogniztion, index of the root note in chromatic scale array, CrossEntropyLoss for single label
    # For bass recogniztion, index of the bass note in chromatic scale array, CrossEntropyLoss for single label

model = ChordCNN(chromatic_notes=12).to(device)
# pos_weight addresses class imbalance in note detection
# A chord with 4 notes has 8 absent notes - without weighting the model
# can get 67% note accuracy by predicting everything absent
# pos_weight=2.0 means false negatives are penalized 2x more than false positives
pos_weight   = torch.tensor([2.0]).to(device)
note_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
root_criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # Label smoothing to help overconfidence and hopefully improve generalizing outside
bass_criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # Label smoothing to help overconfidence and hopefully improve generalizing outside
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# Reduce learning rate by 50% if exact match doesn't improve for 5 epochs
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=8
)

def note_accuracy(note_out, note_labels, threshold=THRESHOLD):
    # Percentage of individual note predictions correct
    preds = (torch.sigmoid(note_out) >= threshold).float()
    correct = (preds == note_labels).float()
    return correct.mean().item()

def note_precision_recall(note_out, note_labels, threshold=THRESHOLD):
    # Precision: of notes predicted present, how many actually are?
    # Recall: of notes actually present, how many did we detect?
    # Low recall = missing notes (Dminor9 F problem)
    # Low precision = hallucinating extra notes
    preds     = (torch.sigmoid(note_out) >= threshold).float()
    tp        = (preds * note_labels).sum()
    precision = (tp / (preds.sum() + 1e-8)).item()
    recall    = (tp / (note_labels.sum() + 1e-8)).item()
    return precision, recall

def exact_match(note_out, note_labels, threshold=THRESHOLD):
    # Percentage of samples where ALL 12 notes predicted correctly (for note recogniztion)
    preds = (torch.sigmoid(note_out) >= threshold).float()
    all_correct = (preds == note_labels).all(dim=1).float()
    return all_correct.mean().item()

def root_accuracy(root_out, root_labels):
    # Percentage of root notes predicted correctly
    predicted = root_out.argmax(dim=1)
    return (predicted == root_labels).float().mean().item()

def bass_accuracy(bass_out, bass_labels):
    # Percentage of bass notes predicted correctly
    predicted = bass_out.argmax(dim=1)
    return (predicted == bass_labels).float().mean().item()

def train(epoch):
    model.train()

    losses, note_accs, exact_accs, root_accs, bass_accs = [], [], [], [], []
    precisions, recalls = [], []

    for spectrograms, note_labels, root_labels, bass_labels in train_loader:

        # Move data to device
        spectrograms = spectrograms.to(device)
        note_labels  = note_labels.to(device)
        root_labels  = root_labels.to(device)
        bass_labels  = bass_labels.to(device)

        optimizer.zero_grad()
        note_out, root_out, bass_out = model(spectrograms)

        note_loss = note_criterion(note_out, note_labels)
        root_loss = root_criterion(root_out, root_labels)
        bass_loss = bass_criterion(bass_out, bass_labels)
        loss = note_loss + ROOT_LOSS_WEIGHT * root_loss + BASS_LOSS_WEIGHT * bass_loss

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        note_accs.append(note_accuracy(note_out, note_labels))
        exact_accs.append(exact_match(note_out, note_labels))
        root_accs.append(root_accuracy(root_out, root_labels))
        bass_accs.append(bass_accuracy(bass_out, bass_labels))
        p, r = note_precision_recall(note_out, note_labels)
        precisions.append(p)
        recalls.append(r)

    print(f'Epoch {epoch+1}/{epochs} | '
          f'Loss: {sum(losses)/len(losses):.4f} | '
          f'Exact Match: {sum(exact_accs)/len(exact_accs):.4f} | '
          f'Root Acc: {sum(root_accs)/len(root_accs):.4f} | '
          f'Precision: {sum(precisions)/len(precisions):.4f} | '
          f'Recall: {sum(recalls)/len(recalls):.4f}')

def test():
    losses, note_accs, exact_accs, root_accs, bass_accs = [], [], [], [], []
    precisions, recalls = [], []

    with torch.no_grad():
        for spectrograms, note_labels, root_labels, bass_labels in test_loader:

            # Move data to device
            spectrograms = spectrograms.to(device)
            note_labels  = note_labels.to(device)
            root_labels  = root_labels.to(device)
            bass_labels  = bass_labels.to(device)

            note_out, root_out, bass_out = model(spectrograms)

            note_loss = note_criterion(note_out, note_labels)
            root_loss = root_criterion(root_out, root_labels)
            bass_loss = bass_criterion(bass_out, bass_labels)
            loss = note_loss + ROOT_LOSS_WEIGHT * root_loss + BASS_LOSS_WEIGHT * bass_loss

            losses.append(loss.item())
            note_accs.append(note_accuracy(note_out, note_labels))
            exact_accs.append(exact_match(note_out, note_labels))
            root_accs.append(root_accuracy(root_out, root_labels))
            bass_accs.append(bass_accuracy(bass_out, bass_labels))
            p, r = note_precision_recall(note_out, note_labels)
            precisions.append(p)
            recalls.append(r)

    avg_exact = sum(exact_accs) / len(exact_accs)
    avg_root  = sum(root_accs)  / len(root_accs)
    avg_bass  = sum(bass_accs)  / len(bass_accs)

    print(f'Test | '
          f'Loss: {sum(losses)/len(losses):.4f} | '
          f'Exact Match: {avg_exact:.4f} | '
          f'Root Acc: {avg_root:.4f} | '
          f'Precision: {sum(precisions)/len(precisions):.4f} | '
          f'Recall: {sum(recalls)/len(recalls):.4f}')

    # Step scheduler based on root_accuracy - more stable signal than exact_match
    # exact_match requires all 12 notes correct so one extra note = full failure (very noisy)
    # root_accuracy is a single categorical prediction, much smoother signal
    scheduler.step(avg_root)

    return avg_exact, avg_root, avg_bass


if __name__ == '__main__':

    best_acc = 0.0

    for epoch in range(epochs):
        train(epoch)
        exact_acc, root_acc, bass_acc = test()

        if float(exact_acc) > best_acc:
            best_acc = exact_acc
            torch.save(model.state_dict(), path)
            print(f'  New best model saved | '
                  f'Exact Match: {best_acc:.4f} | '
                  f'Root Acc: {root_acc:.4f} | '
                  f'Bass Acc: {bass_acc:.4f}')

    print(f'\nTraining complete. Best exact match: {best_acc:.4f}')