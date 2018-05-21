# StarCraft1_RL_Multiagent  

## Description  
__This work bases on my master's thesis.__  
we propose a hierarchical DDPG algorithm to solve the Multi-Agent cooperation problem
__Main task:__ StarCraft1 small scale combat(unit micro control)  
__Mian problem:__ Multi-Agent cooperation problem  
__Performance:__  
Voide link to youku  
[![youku](https://github.com/Kyle1993/StarCraft1_RL_multiagent/blob/master/Screenshot.png =10x10)](http://v.youku.com/v_show/id_XMzYxODMyNzA1Mg==.html?spm=a2hzp.8244740.0.0)
Voide link to youtube
[![youku](https://github.com/Kyle1993/StarCraft1_RL_multiagent/blob/master/Screenshot.png){:height="30%" width="30%"}](http://v.youku.com/v_show/id_XMzYxODMyNzA1Mg==.html?spm=a2hzp.8244740.0.0)


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
