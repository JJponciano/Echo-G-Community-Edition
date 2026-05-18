#Command
```bash
sudo apt install falkon -y
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3.5:2b
curl -fsSL https://openclaw.ai/install.sh | bash
echo 'export PATH="$HOME/.npm-global/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
ollama launch openclaw --model qwen3.5:2b --yes
```
