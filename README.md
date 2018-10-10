# StarCraft1_RL_Multiagent  

## Description  
__This work bases on my master's thesis.__  
we propose a hierarchical DDPG algorithm to solve the Multi-Agent cooperation problem  
__Main task:__ StarCraft1 small scale combat(unit micro control)  
__Mian problem:__ Multi-Agent cooperation problem  
__Performance:__  
Voide link of youku:  
[![youku](https://github.com/Kyle1993/StarCraft1_RL_multiagent/blob/master/Screenshot.png)](http://v.youku.com/v_show/id_XMzg1Njk0OTY2NA==.html?spm=a2h3j.8428770.3416059.1)


## Game enveriment
[gym-starcraft-modified](https://github.com/Kyle1993/gym-starcraft-modified)  
It's depends on TorchCraft, I warp an SimpleBattleEnv for multi_units battle, you can follow the installation above.

## Requirement
pytorch  
gym  

## train
* remember change the paramater(`MYSELF_NUM`, `ENEMY_NUM`) to fit the map setting  
* model will be saved every SAVE_ITERVAL episodes
```
python3 sc1_train_hierarchical.py
```

## test
* change the `saved_floder` and `episode_num` to chose the model you want to load  
```
python3 test hiernet.py
```
