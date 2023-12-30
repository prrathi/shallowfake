pip cache purge
python3 -m pip uninstall -y -r <(pip freeze);
cd robust_speech; python3 -m pip install .; cd ..;
python3 -m pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
python3 -m pip install -r requirements.txt;
# may need to install torchaudio from source. if so uncomment and run these two lines:
# wget https://download.pytorch.org/whl/cu111/torchaudio-0.10.0%2Bcu111-cp39-cp39-linux_x86_64.whl#sha256=098269fb1b901269009a7ce137fc84a1cb4689e12aa0111c19879201f00a0b9f
# python3 -m pip install torchaudio-0.10.0+cu111-cp39-cp39-linux_x86_64.whl