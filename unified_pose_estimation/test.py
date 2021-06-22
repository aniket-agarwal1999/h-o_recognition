from tqdm import tqdm
import torch

import numpy as np
import os

from cfg import parameters
from net import UnifiedNetwork, InteractionRNN
from dataset import UnifiedPoseDataset
from visualize import UnifiedVisualization
from argparse import ArgumentParser


parser = ArgumentParser(description='UnifiedPoseEstimation')
parser.add_argument('--mode', type=int, choices=[1, 2], default=1)
parser.add_argument('--normalized', type=bool, default=False)
parser.add_argument('--ckpt_uni', type=str, default='../models/unified_net.pth')
parser.add_argument('--ckpt_lstm', type=str, default='../models/lstm_net.pth')
parser.add_argument('--visualize', type=bool, default=False)
args = parser.parse_args()

training_dataset = UnifiedPoseDataset(mode='test', loadit=True, name='test', normalized=args.normalized)    ### This is testing dataset only
training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size = 1, shuffle=False, num_workers=1)

if args.mode == 1:

    model = UnifiedNetwork()
    model.load_state_dict(torch.load(args.ckpt_uni))
    model.eval()
    model.cuda()

    #optimizer = torch.optim.Adam(model.parameters(), lr=parameters.lr)
        
    # validation

    with torch.no_grad():

        hand_cell_counter = 0.
        object_cell_counter = 0.
        object_counter = 0.
        action_counter = 0.
        interaction_cell_counter = 0.
        interaction_counter = 0.

        hand_detected = False
        object_detected = False

        for batch, data in enumerate(tqdm(training_dataloader)):
            example = training_dataset.samples[batch]
            print(example)
            image = data[0]
            # print('image shape: ', image.shape)
            true = [x.cuda() for x in data[1:]]

            pred = model(image.cuda())
            # loss = model.total_loss(pred, true)
            
            #print loss.data.cpu().numpy()

            pred_hand_pose, pred_action_prob, pred_hand_conf, pred_object_pose, pred_object_prob, pred_object_conf = [p.data.cpu().numpy() for p in pred]
            true_hand_pose, true_action_prob, hand_mask, true_object_pose, true_object_prob, object_mask = [t.data.cpu().numpy() for t in true]

            # print('Pred_hand_conf range: ', pred_hand_conf.min(), ' - ', pred_hand_conf.max())
            # print('Pred_hand_conf mean: ', pred_hand_conf.mean())
            # print('Pred_hand_pose shape: ', pred_hand_pose.shape)
            # print('pred_action_prob shape: ', pred_action_prob.shape)
            # print('pred_hand_conf shape: ', pred_hand_conf.shape)
            # print('pred_object_pose shape: ', pred_object_pose.shape)
            # print('pred_object_prob shape: ', pred_object_prob.shape)
            # print('pred_object_conf shape: ', pred_object_conf.shape)


            #### Basically will give the cell where the highest confidence is found and hence that would be the cell
            true_hand_cell = np.unravel_index(hand_mask.argmax(), hand_mask.shape)
            true_object_cell = np.unravel_index(object_mask.argmax(), object_mask.shape)


            ##### This will give the thing as above but just the predicted one
            pred_hand_cell = np.unravel_index(pred_hand_conf.argmax(), pred_hand_conf.shape)
            pred_object_cell = np.unravel_index(pred_object_conf.argmax(), pred_object_conf.shape)
            
            # hand cell correctly detected
            hand_cell_counter += int(true_hand_cell == pred_hand_cell)
            hand_detected = true_hand_cell == pred_hand_cell

            # object cell correctly detected
            object_cell_counter += int(true_object_cell == pred_object_cell)
            object_detected = true_object_cell == pred_object_cell

            #### For interaction action accuracy
            if hand_detected and object_detected:
                interaction_cell_counter += 1

            z, v, u = true_hand_cell[1:]
            dels = pred_hand_pose[0,:,z, v, u].reshape(21, 3)
            del_u, del_v, del_z = dels[:,0], dels[:,1], dels[:,2]
            hand_points = training_dataset.target_to_control(del_u, del_v, del_z, (u, v, z))
            
            object_increment = 0
            action_increment = 0
            if hand_detected:
                action_increment = int(np.argmax(pred_action_prob[0, :, z, v, u]) == true_action_prob[0, z, v, u])
                action_counter += action_increment

            z, v, u = true_object_cell[1:]
            dels = pred_object_pose[0,:,z, v, u].reshape(21, 3)
            del_u, del_v, del_z = dels[:,0], dels[:,1], dels[:,2]
            object_points = training_dataset.target_to_control(del_u, del_v, del_z, (u, v, z))

            if object_detected:
                object_increment = int(np.argmax(pred_object_prob[0, :, z, v, u]) == true_object_prob[0, z, v, u])
                object_counter += object_increment

            if action_increment and object_increment:
                interaction_counter += 1

            # print training_dataset.samples[batch]

            if args.visualize:
                ### Saving the visualized plots
                if not os.path.isdir(os.path.join('./visuals', example['subject'])):
                    os.mkdir(os.path.join('./visuals', example['subject']))
                if not os.path.isdir(os.path.join('./visuals', example['subject'], example['action_name'])):
                    os.mkdir(os.path.join('./visuals', example['subject'], example['action_name']))
                if not os.path.isdir(os.path.join('./visuals', example['subject'], example['action_name'], example['seq_idx'])):
                    os.mkdir(os.path.join('./visuals', example['subject'], example['action_name'], example['seq_idx']))

                viz = UnifiedVisualization()
                viz.plot_hand(hand_points)
                viz.plot_box(object_points[1:9, :])
                viz.plot_rgb(training_dataset.fetch_image(training_dataset.samples[batch]))
                location = os.path.join('./visuals', example['subject'], example['action_name'], example['seq_idx'], str(example['frame_idx']) + '.png')
                viz.plot(location)


        print(hand_cell_counter * 1. / batch)
        print(object_cell_counter * 1. / batch)
        print(action_counter * 1 / hand_cell_counter)
        print(object_counter * 1 / object_cell_counter)
        print(interaction_counter * 1 / interaction_cell_counter)
else:

    model = UnifiedNetwork()
    model.load_state_dict(torch.load(args.ckpt_uni))
    model.eval()
    model.cuda()

    lstm_model = InteractionRNN()
    lstm_model.load_state_dict(torch.load(args.ckpt_lstm))
    lstm_model.eval()
    lstm_model.cuda()

    with torch.no_grad():

        hand_cell_counter = 0.
        object_cell_counter = 0.
        object_counter = 0.
        action_counter = 0.
        interaction_cell_counter = 0.
        interaction_counter = 0.

        hand_detected = False
        object_detected = False

        prev_idx = 0
        prev_action = 'random'
        for batch, data in enumerate(tqdm(training_dataloader)):
            example = training_dataset.samples[batch]
            print(example)
            image = data[0]
            # print('image shape: ', image.shape)
            true = [x.cuda() for x in data[1:]]

            pred = model(image.cuda())
            # loss = model.total_loss(pred, true)
            
            #print loss.data.cpu().numpy()

            pred_hand_pose, pred_action_prob, pred_hand_conf, pred_object_pose, pred_object_prob, pred_object_conf = [p.data.cpu().numpy() for p in pred]
            true_hand_pose, true_action_prob, hand_mask, true_object_pose, true_object_prob, object_mask = [t.data.cpu().numpy() for t in true]


            #### Basically will give the cell where the highest confidence is found and hence that would be the cell
            true_hand_cell = np.unravel_index(hand_mask.argmax(), hand_mask.shape)
            true_object_cell = np.unravel_index(object_mask.argmax(), object_mask.shape)


            ##### This will give the thing as above but just the predicted one
            pred_hand_cell = np.unravel_index(pred_hand_conf.argmax(), pred_hand_conf.shape)
            pred_object_cell = np.unravel_index(pred_object_conf.argmax(), pred_object_conf.shape)
            
            # hand cell correctly detected
            hand_cell_counter += int(true_hand_cell == pred_hand_cell)
            hand_detected = true_hand_cell == pred_hand_cell

            # object cell correctly detected
            object_cell_counter += int(true_object_cell == pred_object_cell)
            object_detected = true_object_cell == pred_object_cell

            # #### For interaction action accuracy
            # if hand_detected and object_detected:
            #     interaction_cell_counter += 1

            z, v, u = true_hand_cell[1:]
            hand_pose = torch.tensor(pred_hand_pose[0, :, z, v, u])
            gt_action_cell = torch.tensor(true_action_prob[0, z, v, u]).unsqueeze(0)
            dels = pred_hand_pose[0,:,z, v, u].reshape(21, 3)
            del_u, del_v, del_z = dels[:,0], dels[:,1], dels[:,2]
            hand_points = training_dataset.target_to_control(del_u, del_v, del_z, (u, v, z))

            z, v, u = true_object_cell[1:]
            object_pose = torch.tensor(pred_object_pose[0, :, z, v, u])
            gt_object_cell = torch.tensor(true_object_prob[0, z, v, u]).unsqueeze(0)
            dels = pred_object_pose[0,:,z, v, u].reshape(21, 3)
            del_u, del_v, del_z = dels[:,0], dels[:,1], dels[:,2]
            object_points = training_dataset.target_to_control(del_u, del_v, del_z, (u, v, z))

            #### Using LSTM network here
            h_o_pose = torch.cat((hand_pose.unsqueeze(0), object_pose.unsqueeze(0)), 1).cuda()     #### This will be (1, 126) dimensional

            idx = example['seq_idx']
            action = example['action_name']
            if prev_idx != idx or prev_action != action:
                if batch > 0: 
                    pred = lstm_model(new_tensor)         ### new_tensor shape: (seq_len, 126)
                    pred_action, pred_object = pred
                    pred_action = pred_action.cpu().numpy()
                    pred_object = pred_object.cpu().numpy()
                    gt_action = gt_action.numpy()
                    gt_object = gt_object.numpy()

                    hand_detection_li = np.array(hand_detection_li)
                    object_detection_li = np.array(object_detection_li)
                    temp = np.argmax(pred_action, 1) == gt_action
                    action_counter += (temp*hand_detection_li).sum()
                    temp = np.argmax(pred_object, 1) == gt_object
                    object_counter += (temp*object_detection_li).sum()

                new_tensor = h_o_pose
                gt_action = gt_action_cell
                gt_object = gt_object_cell
                prev_idx = idx
                prev_action = action
                hand_detection_li = []
                hand_detection_li.append(hand_detected)
                object_detection_li = []
                object_detection_li.append(object_detected)
            else:
                new_tensor = torch.cat((new_tensor, h_o_pose), 0)
                gt_action = torch.cat((gt_action, gt_action_cell), 0)
                gt_object = torch.cat((gt_object, gt_object_cell), 0)
                hand_detection_li.append(hand_detected)
                object_detection_li.append(object_detected)
            
            # object_increment = 0
            # action_increment = 0
            # if hand_detected:
            #     action_increment = int(np.argmax(pred_action_prob[0, :, z, v, u]) == true_action_prob[0, z, v, u])
            #     action_counter += action_increment


            # if object_detected:
            #     object_increment = int(np.argmax(pred_object_prob[0, :, z, v, u]) == true_object_prob[0, z, v, u])
            #     object_counter += object_increment

            # if action_increment and object_increment:
            #     interaction_counter += 1

            # print training_dataset.samples[batch]

            if args.visualize:
                ### Saving the visualized plots
                if not os.path.isdir(os.path.join('./visuals', example['subject'])):
                    os.mkdir(os.path.join('./visuals', example['subject']))
                if not os.path.isdir(os.path.join('./visuals', example['subject'], example['action_name'])):
                    os.mkdir(os.path.join('./visuals', example['subject'], example['action_name']))
                if not os.path.isdir(os.path.join('./visuals', example['subject'], example['action_name'], example['seq_idx'])):
                    os.mkdir(os.path.join('./visuals', example['subject'], example['action_name'], example['seq_idx']))

                viz = UnifiedVisualization()
                viz.plot_hand(hand_points)
                viz.plot_box(object_points[1:9, :])
                viz.plot_rgb(training_dataset.fetch_image(training_dataset.samples[batch]))
                location = os.path.join('./visuals', example['subject'], example['action_name'], example['seq_idx'], str(example['frame_idx']) + '.png')
                viz.plot(location)


        print(hand_cell_counter * 1. / batch)
        print(object_cell_counter * 1. / batch)
        print(action_counter * 1 / hand_cell_counter)
        print(object_counter * 1 / object_cell_counter)
        # print(interaction_counter * 1 / interaction_cell_counter)
    