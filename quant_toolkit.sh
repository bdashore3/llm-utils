#!/bin/bash

# Remove old dependencies and install/clone new ones
setup() {
    # Install nano and python alias for easy running
    apt update
    apt install -y nano python-is-python3

    # Install dev versions of transformers and auto GPTQ
    pip uninstall -y auto-gptq transformers
    pip install git+https://github.com/huggingface/transformers https://github.com/PanQiWei/AutoGPTQ/releases/download/v0.3.2/auto_gptq-0.3.2+cu118-cp310-cp310-linux_x86_64.whl

    # Get Bloke's scripts for quanting and HF model downloading (if needed)
    wget https://gist.github.com/TheBloke/b47c50a70dd4fe653f64a12928286682/raw/ebcee019d90a178ee2e6a8107fdd7602c8f1192a/quant_autogptq.py
    wget https://gist.github.com/TheBloke/b2302eb2bc1fd58359d6f9d54d57684a/raw/f02c4e5a4b160397d32b2adf6cf969f6d8df2fe2/hub_download.py

    config_vars
}

config_vars() {
    read -r -p "Model repo? (user/model name): " MODEL_REPO
    read -r -p "Quantized model repo? " QUANT_MODEL_REPO

    read -r -p "Github username?" GITHUB_USERNAME
    git config --global user.name "$GITHUB_USERNAME"

    read -r -p "Github email?" GITHUB_EMAIL
    git config --global user.email "$GITHUB_EMAIL"
}

fetch_model() {
    MODEL_FOLDER=$("$MODEL_REPO" | cut -d "/" -f 2)
    exec "python hub_download.py $MODEL_REPO $MODEL_FOLDER"
}

# Runs a quantization. Parameters provided as arguments
# 1. group size, 2. act order, 3. dtype
run_quant() {
    if [ "$2" != 1 ] && [ "$2" != 0 ]; then
        echo "Invalid act order was specified"
        exit 1
    fi;

    quant_folder_suffix=""
    case $1 in
        32)
            quant_folder_suffix="-32g-actorder"
            QUANT_BRANCH="4bit-32g-actorder"
            ;;

        128)
            if [ "$2" == 1 ]; then
                quant_folder_suffix="-128g-actorder"
                QUANT_BRANCH="4bit-128g-actorder"
            elif [ "$2" == 0 ]; then
                quant_folder_suffix=""
                QUANT_BRANCH="main"
            else
                echo "Invalid act order was specified"
                exit 1
            fi;
            ;;

        *)
            echo "Invalid groupsize was specified"
            exit 1
            ;;
    esac

    QUANT_FOLDER="${QUANT_MODEL_REPO}${quant_folder_suffix}"

    if ! (git clone "https://huggingface.co/$QUANT_MODEL_REPO" "$QUANT_FOLDER"); then
        echo "Git clone of quantized repo failed. Make sure it's created in HuggingFace!"
        exit 1
    fi;

    huggingface-cli lfs-enable-largefiles "$QUANT_FOLDER"

    while true; do
        read -r -p "Are you sure? This model will be quantized with a group size of $1, act_order of $1, and a dtype of $3 (y/n)" confirm_quant
        case $confirm_quant in
            [Yy]* )
                exec "python quant_autogptq.py $MODEL_FOLDER $QUANT_FOLDER wikitext --bits 4 --group_size $1 --desc_act $2 --damp 0.01 --dtype $3 --seqlen 4096 --num_samples 128 --use_triton --use_fast --cache_examples 1"
                ;;
            [Nn]* )
                exit
                ;;
            * ) 
                echo "Please answer y or n."
                ;;
        esac
    done
}

add_and_push() {
    cd "$QUANT_FOLDER" || return

    if [ "$QUANT_BRANCH" != "main" ]; then
        git checkout -b "$QUANT_BRANCH"
    fi;

    git add .
    git commit -m "Initial GPTQ model commit"
    exec "git push -u origin $QUANT_BRANCH"
}

cleanup_vars() {
    MODEL_REPO=""
    QUANT_MODEL_REPO=""
    quant_folder_suffix=""
    QUANT_FOLDER=""
    GITHUB_USERNAME=""
    GITHUB_EMAIL=""

    huggingface-cli logout
    git config --global user.email ""
    git config --global user.name ""
}
