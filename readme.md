```
Proximos passos:
- Tutorial de Stable baselines3 do @sentdex: 
- - https://www.youtube.com/watch?v=XbWhJdQgi7E&ab_channel=sentdex
- - https://pythonprogramming.net/introduction-reinforcement-learning-stable-baselines-3-tutorial/
- SB3 + Starcraft2 (rodar no Windows): https://www.youtube.com/watch?v=q59wap1ELQ4&ab_channel=sentdex
- (Vou tentar apenas pelos tutorias do @sentdex, primeiro) Serie de aula sobre Reinforcement Learning: https://www.youtube.com/watch?v=2pWv7GOvuf0&ab_channel=GoogleDeepMind
```

```
Doc:

- Stable-baselines3: https://stable-baselines3.readthedocs.io/en/master/
- - https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
- - https://stable-baselines3.readthedocs.io/en/master/guide/quickstart.html

- [Entendi que esta DEPRECATED] Stable-baselines: https://stable-baselines.readthedocs.io/en/master/

- Artigo no Medium: https://towardsdatascience.com/stable-baselines-a-fork-of-openai-baselines-reinforcement-learning-made-easy-df87c4b2fc82


```

```
Esses dois links, me parecem bons tutorias. Não vi ainda:
- https://spinningup.openai.com/en/latest/
- https://huggingface.co/learn/deep-rl-course/unit0/introduction
```

```
Esse tutorial parece muito bom: https://community.ops.io/akilesh/reinforcement-learning-in-super-mario-bros-48a
O unico problema, foi a questão do seed, mostrado em algum
comentario mais a baixo aqui.
```


```
30-03-2024 05:57

main3.py esta rodando. Plotando visualmente, a tentativa de equilibrar uma barra

main2.py esta rodando, mas plotando apenas informações no terminal


main.py esta rodando. No final gera um video do 
jogo a tentativa de equilibrar uma barra.
Chegou salvar o video uma vez. Não sei se continua.
Mas não vou testar novamente no momento, pois
não e algo que seja o foco agora


mario.py esta rodando. Executa visualmente o jogo do mario
Com ações aleatorias. Não ta usando IA (base lines)


marioV2.py. Falhando com o Error:
marioV2doSite.py. Falhando com o Error:
  logger.warn(
/home/gerson-aguiar/.pyenv/versions/3.9.9/lib/python3.9/site-packages/stable_baselines3/common/vec_env/patch_gym.py:49: UserWarning: You provided an OpenAI Gym environment. We strongly recommend transitioning to Gymnasium environments. Stable-Baselines3 is automatically wrapping your environments in a compatibility layer, which could potentially cause issues.
  warnings.warn(
Traceback (most recent call last):
  File "/opt/loggi/gerssonmg/stable_baselines3/marioV2.py", line 24, in <module>
    state = env.reset()
  File "/home/gerson-aguiar/.pyenv/versions/3.9.9/lib/python3.9/site-packages/stable_baselines3/common/vec_env/vec_frame_stack.py", line 41, in reset
    observation = self.venv.reset()
  File "/home/gerson-aguiar/.pyenv/versions/3.9.9/lib/python3.9/site-packages/stable_baselines3/common/vec_env/dummy_vec_env.py", line 77, in reset
    obs, self.reset_infos[env_idx] = self.envs[env_idx].reset(seed=self._seeds[env_idx], **maybe_options)
  File "/home/gerson-aguiar/.pyenv/versions/3.9.9/lib/python3.9/site-packages/shimmy/openai_gym_compatibility.py", line 106, in reset
    return self.gym_env.reset(seed=seed, options=options)
  File "/home/gerson-aguiar/.pyenv/versions/3.9.9/lib/python3.9/site-packages/gym/core.py", line 379, in reset
    obs, info = self.env.reset(**kwargs)
TypeError: reset() got an unexpected keyword argument 'seed'


Entendi que e alguma incompatibilidade entre o gym_mario com a Stable baselines3.
Mas que isso não atrapalha tentar implementar a baselines3 ou gym em outros
cenarios
```

pyenv virtualenv 3.9 stable3

pyenv activate stable3
pip install --upgrade pip

```
Configurando setup, para usar a versao 3

https://stable-baselines.readthedocs.io/en/master/


https://github.com/DLR-RM/stable-baselines3?tab=readme-ov-file

https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/stable_baselines_getting_started.ipynb
```


pip install 'stable-baselines3[extra]'

sudo apt-get install ffmpeg freeglut3-dev xvfb

CUDA_VISIBLE_DEVICES=-1 python main.py



pip install gym-super-mario-bros
pip install nes_py

```
Precisei editar em:
observation, reward, terminated, truncated, info = self.env.step(acti 


Pelo error:
/home/gerson-aguiar/.pyenv/versions/stable3/lib/python3.9/site-packages/gym/utils/passive_env_checker.py:225: DeprecationWarning: `np.bool8` is a deprecated alias for `np.bool_`.  (Deprecated NumPy 1.24)
  if not isinstance(done, (bool, np.bool8)):
Traceback (most recent call last):
  File "/opt/loggi/gerssonmg/stable_baselines3/mario.py", line 17, in <module>
    state, reward, done, info = env.step(env.action_space.sample())
  File "/home/gerson-aguiar/.pyenv/versions/stable3/lib/python3.9/site-packages/nes_py/wrappers/joypad_space.py", line 74, in step
    return self.env.step(self._action_map[action])
  File "/home/gerson-aguiar/.pyenv/versions/stable3/lib/python3.9/site-packages/gym/wrappers/time_limit.py", line 50, in step
    observation, reward, terminated, truncated, info = self.env.step(action)
ValueError: not enough values to unpack (expected 5, got 4)
```

pip install torch torchvision torchaudio


pip install pygame