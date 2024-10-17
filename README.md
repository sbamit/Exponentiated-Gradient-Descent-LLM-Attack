## Change Readme File.
This is a Project that explores the Exponentiated Gradient Descent optimizaiton method to produce adversarial suffix to attack algined Large Language Models.
The method is shown to be effective on Llama-2 chat model with 7 Billion parameters. 

To run pgd script on a number of behaviors(i.e., 20), execute the following command in the shell.
  python run_pgd.py \
    --input_file  "/home/samuel/research/llmattacks/llm-attacks/data/advbench/harmful_behaviors.csv" \
    --output_file "./JSON_Files/PGD_AdvBench_Llama2.jsonl" \
    --model "Llama2" \
    --dataset_name "AdvBench" \
    --num_behaviors 20

Next target is to build similar pipeline for the EGD with Adam optim script.