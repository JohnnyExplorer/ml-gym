import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style

import time

style.use("ggplot")

SIZE = 20
MAX_MOVE = 50

DIRECTION_COUNT = 5

HM_EPISODES = 1000000
SHOW_EVERY =  100000  # how often to play through env visually.

MOVE_PENALTY = 5
ENEMY_PENALTY = 500
FOOD_REWARD = 150

epsilon = 0
EPS_DECAY = 0  # Every episode will be epsilon*EPS_DECAY

start_q_table = 'base.pickle'
save_q_table = 'qtable-rate-1.pickle'
#start_q_table = 'learn-rate-5.pickle'
#save_q_table = 'learn-rate-5.pickle'


#start_q_table = 'base.pickle' # None or Filename
#start_q_table = None
LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_N = 1  # player key in dict
FOOD_N = 2  # food key in dict
ENEMY_N = 3  # enemy key in dict

# the dict!
d = {1: (255, 175, 0),
     2: (0, 255, 0),
     3: (0, 0, 255)}


class Blob:
    def __init__(self):
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)

    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    def action(self, choice):
        '''
        Gives us 4 total movement options. (0,1,2,3)
        '''

        # if choice ==  0:
        #     self.move(x=-1, y=-1)
        # elif choice ==  1:
        #     self.move(x=0, y=-1)
        # elif choice ==  2:
        #     self.move(x=1, y=-1)
        # elif choice ==  3:
        #     self.move(x=-1, y=0)
        # elif choice ==  4:
        #     self.move(x=0, y=0)
        # elif choice ==  5:
        #     self.move(x=1, y=0)
        # elif choice ==  6:
        #     self.move(x=-1, y=1)
        # elif choice ==  7:
        #     self.move(x=0, y=1)
        # elif choice ==  8:
        #     self.move(x=1, y=1)


        if choice ==  0:
            self.move(x=0, y=0)
        elif choice ==  1:
            self.move(x=0, y=-1)
        elif choice ==  2:
            self.move(x=0, y=1)
        elif choice ==  3:
            self.move(x=-1, y=0)
        elif choice ==  4:
            self.move(x=0, y=0)
        elif choice ==  5:
            self.move(x=1, y=0)


    def move(self, x=False, y=False):

        # If no value for x, move randomly
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        # If no value for y, move randomly
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y


        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE-1:
            self.x = SIZE-1
        if self.y < 0:
            self.y = 0
        elif self.y > SIZE-1:
            self.y = SIZE-1


if start_q_table is None:
    print('make q table')
    # initialize the q-table#
    q_table = {}
    for i in range(-SIZE+1, SIZE):
        for ii in range(-SIZE+1, SIZE):
            for iii in range(-SIZE+1, SIZE):
                for iiii in range(-SIZE+1, SIZE):
                    q_table[((i, ii), (iii, iiii))] = [np.random.uniform(-5, 0) for i in range(DIRECTION_COUNT)]
    print('make q table..done')
    
    with open(f"base.pickle", "wb") as f:
        pickle.dump(q_table, f)
    exit()
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)


# can look up from Q-table with: print(q_table[((-9, -2), (3, 9))]) for example

episode_rewards = []
episode_move = []
for episode in range(HM_EPISODES):
    player = Blob()
    food = Blob()
    enemy = Blob()
    if episode % SHOW_EVERY == 0:
        print(f"======on #{episode}, epsilon is {epsilon}=====")
        print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(MAX_MOVE):
        obs = (player-food, player-enemy)
        ran = np.random.random()
        if episode % SHOW_EVERY == 0:
            print('obs ', obs)
        if ran > epsilon:
            episode_move.append(1)
            if episode % SHOW_EVERY == 0:
                print('taking action')
            # GET THE ACTION
            action = np.argmax(q_table[obs])
        else:
            episode_move.append(0)
            if episode % SHOW_EVERY == 0:
                print('random action')
            action = np.random.randint(0, DIRECTION_COUNT)
        # Take the action!

        if episode % SHOW_EVERY == 0:
            print('action: ', action)
        player.action(action)

        #### MAYBE ###
        #enemy.move()
        #food.move()
        ##############

        if player.x == enemy.x and player.y == enemy.y:
            if episode % SHOW_EVERY == 0:
                print('dead')
            reward = -ENEMY_PENALTY
        elif player.x == food.x and player.y == food.y:
            if episode % SHOW_EVERY == 0:
                print('reward')
            reward = FOOD_REWARD
        else:
            if episode % SHOW_EVERY == 0:
                print('ranout')
            reward = -MOVE_PENALTY
        ## NOW WE KNOW THE REWARD, LET'S CALC YO
        # first we need to obs immediately after the move.
        new_obs = (player-food, player-enemy)
        if episode % SHOW_EVERY == 0:
            print('new_obs ', new_obs)
        max_future_q = np.max(q_table[new_obs])
        if episode % SHOW_EVERY == 0:
            print('max_future_q ', max_future_q)
        current_q = q_table[obs][action]
        if episode % SHOW_EVERY == 0:
            print('current_q ', current_q)

        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
            if episode % SHOW_EVERY == 0:
                print('new q is food reward')
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            if episode % SHOW_EVERY == 0:
                print('calculating new q')
            if episode % SHOW_EVERY == 0:
                print('calced new q ' , new_q)
        q_table[obs][action] = new_q

        if show == 'none':
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
            env[food.x][food.y] = d[FOOD_N]  # sets the food location tile to green color
            env[player.x][player.y] = d[PLAYER_N]  # sets the player tile to blue
            env[enemy.x][enemy.y] = d[ENEMY_N]  # sets the enemy location to red
            img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
            img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
            cv2.imshow("image", np.array(img))  # show it!
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:  # crummy code to hang at the end if we reach abrupt end for good reasons or not.
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            time.sleep(.3)
           

        episode_reward += reward
        
        if episode % SHOW_EVERY == 0:
            print('episode_reward ', episode_reward)

        if episode % SHOW_EVERY == 0:
            print('epsilon ', epsilon)
            #print('==========')
        
        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
            break


    #print(episode_reward)
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY


if save_q_table:
    with open(save_q_table, "wb") as f:
        pickle.dump(q_table, f)

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')
print('DONE')
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()
