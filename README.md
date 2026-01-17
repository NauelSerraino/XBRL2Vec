## Activate MLFlow environment

```bash
mlflow server --host 127.0.0.1 --port 5000
```

## Activate TensorBoard

```bash
tensorboard --logdir=runs --port=6006
```

## Example run of an experiment

```bash
python /home/nauel/vscode/XBRL2Vec/src/main.py \
    --model_name "D-Linear FAE" \
    --latent_dim 150 \
    --epochs 10 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --pretrain \
    --num_iterations_pretrain 10
```

## Other experiments
```bash
python /home/nauel/vscode/XBRL2Vec/src/main.py --model_name "RNN FAE" --latent_dim 150 --epochs 10 --batch_size 32 --learning_rate 0.001 --lambda_ortho 0.0001 --pretrain --num_iterations_pretrain 10
python /home/nauel/vscode/XBRL2Vec/src/main.py --model_name "LSTM FAE" --latent_dim 150 --epochs 10 --batch_size 32 --learning_rate 0.001 --lambda_ortho 0.0001 --pretrain --num_iterations_pretrain 10
python /home/nauel/vscode/XBRL2Vec/src/main.py --model_name "GRU FAE" --latent_dim 150 --epochs 10 --batch_size 32 --learning_rate 0.001 --lambda_ortho 0.0001 --pretrain --num_iterations_pretrain 10
python /home/nauel/vscode/XBRL2Vec/src/main.py --model_name "Transformer FAE" --latent_dim 150 --epochs 10 --batch_size 32 --learning_rate 0.001 --lambda_ortho 0.0001 --pretrain --num_iterations_pretrain 10
python /home/nauel/vscode/XBRL2Vec/src/main.py --model_name "Euclidean FAE" --latent_dim 150 --epochs 10 --batch_size 32 --learning_rate 0.001 --lambda_ortho 0.0001 --pretrain --num_iterations_pretrain 10
```