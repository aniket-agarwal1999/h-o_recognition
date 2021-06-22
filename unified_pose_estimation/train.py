import random

from tqdm import tqdm
import torch
import numpy as np

from cfg import parameters
from net import UnifiedNetwork, InteractionRNN
from dataset import UnifiedPoseDataset
from argparse import ArgumentParser

from tensorboardX import SummaryWriter


parser = ArgumentParser(description='UnifiedPoseEstimation')
parser.add_argument('--mode', type=int, choices=[1, 2], default=1)
parser.add_argument('--normalized', type=bool, default=False)
parser.add_argument('--ckpt_uni', type=str, default='../models/unified_net.pth')
parser.add_argument('--ckpt_lstm', type=str, default='../models/lstm_net.pth')
parser.add_argument('--tfboard_uni', type=str, default='./tfboards/unified_net/')
parser.add_argument('--tfboard_lstm', type=str, default='./tfboards/lstm_net/')
args = parser.parse_args()

training_dataset = UnifiedPoseDataset(mode='train', loadit=True, name='train', normalized=args.normalized)
testing_dataset = UnifiedPoseDataset(mode='test', loadit=True, name='test', normalized=args.normalized)


if args.mode == 1:

    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size = parameters.batch_size, shuffle=True, num_workers=4)
    testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size = parameters.batch_size, shuffle=False, num_workers=4)

    model = UnifiedNetwork()
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=parameters.lr)

    best_loss = float('inf')

    writer = SummaryWriter(args.tfboard_uni)

    for epoch in range(parameters.epochs):
        
        # train
        
        model.train()
        training_loss = 0.
        for batch, data in enumerate(tqdm(training_dataloader)):

            optimizer.zero_grad()

            image = data[0]
            true = [x.cuda() for x in data[1:]]

            if torch.isnan(image).any():
                raise ValueError('WTF?!')

            pred = model(image.cuda())
            loss = model.total_loss(pred, true)
            training_loss += loss.data.cpu().numpy()  
            loss.backward()

            optimizer.step()
        
        training_loss = training_loss / batch
        # writer.add_scalars('data/loss', {'train_loss': training_loss}, epoch)

        # validation
        if epoch%2 == 0:
            model.eval()
            validation_loss = 0.
            with torch.no_grad():
                for batch, data in enumerate(tqdm(testing_dataloader)):

                    image = data[0]
                    true = [x.cuda() for x in data[1:]]

                    if torch.isnan(image).any():
                        raise ValueError('WTF?!')

                    pred = model(image.cuda())
                    loss = model.total_loss(pred, true)
                    validation_loss += loss.data.cpu().numpy()

            validation_loss = validation_loss / batch
            writer.add_scalars('Loss', {'train_loss': training_loss, 'val_loss': validation_loss}, epoch)

            if validation_loss < best_loss:

                print ("Old loss: {}, New loss : {}. Saving model to disk.".format(best_loss, validation_loss))
                best_loss = validation_loss

                torch.save(model.state_dict(), args.ckpt_uni)
            
            print ("Epoch : {} finished. Training Loss: {}. Validation Loss: {}".format(epoch, training_loss, validation_loss))

else:

    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size = 1, shuffle=False, num_workers=4)      ### shuffle=False is necessary for taking in as sequence
    testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size = 1, shuffle=False, num_workers=4)

    model = InteractionRNN()
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=parameters.lr)

    best_loss = float('inf')

    writer = SummaryWriter(args.tfboard_lstm)

    ### For calling the previous model
    pred_model = UnifiedNetwork()
    pred_model.load_state_dict(torch.load(args.ckpt_uni))
    pred_model.eval()
    pred_model.cuda()

    for epoch in range(60):

        model.train()
        pred_model.eval()
        training_loss = 0.
        divide_batch = 0
        prev_idx = 0
        prev_action = 'random'
        for batch, data in enumerate(tqdm(training_dataloader)):
            optimizer.zero_grad()

            image = data[0]
            true = [x.cuda() for x in data[1:]]

            if torch.isnan(image).any():
                raise ValueError('WTF?!')

            pred = pred_model(image.cuda())
            pred_hand_pose, pred_action_prob, pred_hand_conf, pred_object_pose, pred_object_prob, pred_object_conf = [p.data.cpu().numpy() for p in pred]
            true_hand_pose, true_action_prob, hand_mask, true_object_pose, true_object_prob, object_mask = true

            pred_hand_cell = np.unravel_index(pred_hand_conf.argmax(), pred_hand_conf.shape)
            pred_object_cell = np.unravel_index(pred_object_conf.argmax(), pred_object_conf.shape)

            z, v, u = pred_hand_cell[1:]
            hand_pose = torch.tensor(pred_hand_pose[0, :, z, v, u])
            pred_action = true_action_prob[0, z, v, u].unsqueeze(0)       ### 1*1
            z, v, u = pred_object_cell[1:]
            object_pose = torch.tensor(pred_object_pose[0, :, z, v, u])
            pred_object = true_object_prob[0, z, v, u].unsqueeze(0)

            h_o_pose = torch.cat((hand_pose.unsqueeze(0), object_pose.unsqueeze(0)), 1).cuda()     #### This will be (1, 126) dimensional


            idx = training_dataset.samples[batch]['seq_idx']
            action = training_dataset.samples[batch]['action_name']
            if prev_idx != idx or prev_action!= action:
                if batch > 0: 
                    # print('New_tensor shape: ', new_tensor.shape)
                    # print('Gt_acion shape: ', gt_action.shape)
                    # print('Gt_object shape: ', gt_object.shape)
                    pred = model(new_tensor)         ### new_tensor shape: (seq_len, 126)
                    loss = model.total_loss(pred, gt_action, gt_object)
                    training_loss += loss.data.cpu().numpy() 
                    loss.backward()
                    optimizer.step()
                    divide_batch += 1

                new_tensor = h_o_pose
                gt_action = pred_action
                gt_object = pred_object
                prev_idx = idx
                prev_action = action
            else:
                new_tensor = torch.cat((new_tensor, h_o_pose), 0)
                gt_action = torch.cat((gt_action, pred_action), 0)
                gt_object = torch.cat((gt_object, pred_object), 0)


        training_loss = training_loss/divide_batch

        # validation
        if epoch%2 == 0:
            model.eval()
            validation_loss = 0.
            divide_batch = 0
            prev_idx = 0
            prev_action = 'random'
            with torch.no_grad():
                for batch, data in enumerate(tqdm(testing_dataloader)):

                    image = data[0]
                    true = [x.cuda() for x in data[1:]]

                    if torch.isnan(image).any():
                        raise ValueError('WTF?!')

                    pred = pred_model(image.cuda())
                    pred_hand_pose, pred_action_prob, pred_hand_conf, pred_object_pose, pred_object_prob, pred_object_conf = [p.data.cpu().numpy() for p in pred]
                    true_hand_pose, true_action_prob, hand_mask, true_object_pose, true_object_prob, object_mask = true

                    pred_hand_cell = np.unravel_index(pred_hand_conf.argmax(), pred_hand_conf.shape)
                    pred_object_cell = np.unravel_index(pred_object_conf.argmax(), pred_object_conf.shape)

                    z, v, u = pred_hand_cell[1:]
                    hand_pose = torch.tensor(pred_hand_pose[0, :, z, v, u])
                    pred_action = true_action_prob[0, z, v, u].unsqueeze(0)       ### 1*1
                    z, v, u = pred_object_cell[1:]
                    object_pose = torch.tensor(pred_object_pose[0, :, z, v, u])
                    pred_object = true_object_prob[0, z, v, u].unsqueeze(0)

                    idx = testing_dataset.samples[batch]['seq_idx']
                    if prev_idx != idx or prev_action!=action:
                        if batch > 0: 
                            # print('New_tensor shape: ', new_tensor.shape)
                            # print('Gt_acion shape: ', gt_action.shape)
                            # print('Gt_object shape: ', gt_object.shape)
                            pred = model(new_tensor)         ### new_tensor shape: (seq_len, 126)
                            loss = model.total_loss(pred, gt_action, gt_object)
                            validation_loss += loss.data.cpu().numpy() 
                            
                            divide_batch += 1

                        new_tensor = h_o_pose
                        gt_action = pred_action
                        gt_object = pred_object
                        prev_idx = idx
                        prev_action = action
                    else:
                        new_tensor = torch.cat((new_tensor, h_o_pose), 0)
                        gt_action = torch.cat((gt_action, pred_action), 0)
                        gt_object = torch.cat((gt_object, pred_object), 0)

            validation_loss = validation_loss / batch
            writer.add_scalars('Loss', {'train_loss': training_loss, 'val_loss': validation_loss}, epoch)

            if validation_loss < best_loss:

                print ("Old loss: {}, New loss : {}. Saving model to disk.".format(best_loss, validation_loss))
                best_loss = validation_loss

                torch.save(model.state_dict(), args.ckpt_lstm)

            print ("Epoch : {} finished. Training Loss: {}. Validation Loss: {}".format(epoch, training_loss, validation_loss))
