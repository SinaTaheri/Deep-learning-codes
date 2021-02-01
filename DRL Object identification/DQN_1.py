import torch
import torch.nn as nn
import numpy as np
import random

# parameter
gamma = 0.1
epsilon_min = 0.1 # minimum exploration value (the smaller the more exploitation)
epsilon_max = 1.0 # maximum exploration value (the larger the more exploration)
epsilon = epsilon_max
epsilon_dec_steps = 5
epsilon_dec = (epsilon_max - epsilon_min) / epsilon_dec_steps
target_update_interval = 1000
max_step = 10000
experience = [] # experience replay
experience_buffer_size = 2000
experience_sample_size = 15

# class for defining a state
class State:
    def __init__(self, input_tensor):
        self.input_tensor = input_tensor

    def state_feature(self):
        return self.input_tensor


# Q_learning model
def DQL_model(input=1024, output=9):
    """
    Initialize and return a simple discriminator model.
    """
    model = torch.nn.Sequential(torch.nn.Linear(in_features=input, out_features=1024),
                                torch.nn.ReLU(),
                                torch.nn.Linear(in_features=1024, out_features=output)
            )
    return model


# calculate q_values of given state input
def get_q_values(state, model):
    input_features = state
    predict_q_values = model(input_features)
    q_values = predict_q_values.detach().numpy()
    return q_values


# select action base on q_values of current state
# there are 9 possible action indexed from 0 -> 8
def select_action(epsilon, action_values):
    # do exploitation if epsilon less than random in range (0,1)
    if random.random() > epsilon:
        action = np.argmax(action_values)

    # do exploration if epsilon greater than random in range (0,1)
    # generate random action index between 0 and 8
    else:
        action = random.randint(0,9)

    return action


# place holder for take_action function need Sina and Arhm function for take_action
def take_action(bbox, action, stopping_condition):
    #using the initial_bbox to transform the image using a function
    ############
    # reward is the confidence score
    next_state_input_tensor, reward, next_bbox, current_epoch= [] #Ahm function return
    if reward == 1:
        stopping_condition = True
    next_state = State(next_state_input_tensor)
    return reward, next_state, next_bbox, stopping_condition, current_epoch
##########################################################


# compute next state in target network
def compute_target(r, next_state, target_model):
    return r + gamma * np.amax(get_q_values(next_state.state_feature(), target_model))


# using experience inside experience_replay to train the model
def apply_experience(experience, Qmodel, target_model):
     random.shuffle(experience)
     if len(experience) < experience_sample_size:
         sample_size = len(experience)
     else:
         sample_size = experience_sample_size
     for random_exp in experience[0:sample_size]:
         current_state, action_values, action, state_reward, next_state, stopping_condition = random_exp
         # compute q_values of next state at target model
         target_value = compute_target(state_reward, next_state, target_model)
         if stopping_condition != True:
             # Compute and print loss
             loss = criterion(action_values[action], target_value)
             # Zero gradients, perform a backward pass, and update the weights.
             optimizer.zero_grad()
             loss.backward()
             optimizer.step()


#update the target_model with weight from Qmodel
def update_target_model(Qmodel, target_model):
    target_model.load_state_dict(Qmodel.state_dict())


# initiate model, loss function and optimizer
Qmodel =  DQL_model()
target_model = DQL_model()
criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(Qmodel.parameters(), lr=1e-4)


# training function for deep Q network, this return the current epsilon to keep track
def DQL_train (state_features, Qmodel, target_model, box_coor, epsilon=epsilon_max):

    # coordinate of the box in list format [x1,y1,x2,y2]
    initial_bbox = box_coor
    bbox = initial_bbox
    # using the initial_bbox to transform the image using a function
    # Sina function
    ############
    step = 0
    total_step = 0
    # create initial state base on parsing state_features
    current_state = State(state_features)
    stopping_condition = False

    # keep exploring the original image until either max_step is reached or the object has been identified
    while step != max_step or stopping_condition != True:
        # compute q values and decide the action base on q values of the current state
        action_values = get_q_values(current_state.state_feature(), Qmodel)
        action = select_action(epsilon, action_values)
        # apply action to the current state to get the next state and reward and check for stopping condition and epoch#
        state_reward, next_state, next_bbox, stopping_condition, current_epoch= take_action(bbox, action,stopping_condition)
        step_experience = (current_state, action_values, action, state_reward, next_state, stopping_condition)

        # save the state action sequence into experience replay
        experience.append(step_experience)

        # apply experience replay
        apply_experience(experience,Qmodel,target_model)

        # move to the next state
        current_state = next_state
        bbox = next_bbox
        step += 1
        total_step += 1

        # if reaching the step update number (10) then update target model with weight from target_model
        if total_step % target_update_interval  == 0:
            update_target_model(Qmodel, target_model)

    # reduce eploration parameter as the we becoming more confident in our action
    if current_epoch < epsilon_dec_steps:
        epsilon -= epsilon_dec

    return epsilon

# testing function for deep Q network after training
def DQL_testing(state_features, box_coor, Qmodel):

    # coordinate of the box in list format [x1,y1,x2,y2]
    initial_bbox = box_coor
    bbox = initial_bbox
    # using the initial_bbox to transform the image using a function
    # Sina function
    ############
    step = 0
    total_step = 0
    # create initial state base on parsing state_features
    current_state = State(state_features)
    stopping_condition = False

    while step != max_step or stopping_condition != True:
        # compute q values and decide the action base on q values of the current state
        action_values = get_q_values(current_state.state_feature(), Qmodel)
        action = select_action(epsilon, action_values)
        # apply action to the current state to get the next state and reward and check for stopping condition and epoch#
        state_reward, next_state, next_bbox, stopping_condition, current_epoch = take_action(bbox, action)

        # move to the next state
        current_state = next_state
        bbox = next_bbox
        step += 1


