#set -x

version=v2

# 定义长选项
OPTIONS=$(getopt -o a --long result_root:,data_indexes_path:,fid_source_path:,generation_checkpoint:,generation_model_config:,version: -- "$@")  # -o ab:c:
eval set -- "$OPTIONS"
while true; do
    case "$1" in
        --result_root)
            result_root="$2"
            shift 2
            ;;
        --data_indexes_path)
            data_indexes_path="$2"
            shift 2
            ;;
        --fid_source_path)
            fid_source_path="$2"
            shift 2
            ;;
        --generation_checkpoint)
            generation_checkpoint="$2"
            shift 2
            ;;
        --generation_model_config)
            generation_model_config="$2"
            shift 2
            ;;
        --segmentation_checkpoint)
            segmentation_checkpoint="$2"
            shift 2
            ;;
        --version)
            version="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
    esac
done

log="result_root: $result_root, \n
data_indexes_path: $data_indexes_path, \n
fid_source_path: $fid_source_path, \n
version: $version"
echo -e $log
mkdir -p ${HOME}/${result_root}
echo -e $log > ${HOME}/${result_root}/eval.log
sleep 10

python3 custom_port.py --show_root ${result_root} \
--data_indexes_path ${data_indexes_path} \
--show --version $version

python3 -m pytorch_fid ${HOME}/${result_root}/result \
${fid_source_path} --report_path ${HOME}/${result_root}


cd ${HOME}/code/ReMOVE
python3 evaluate.py \
  --result_path ${result_root}/result \
  --show_path ${result_root}/evaluation_ReMOVE_crop1 \
  --data_indexes_path $data_indexes_path \
  --crop 1

cd -


python3 ${HOME}/code/unhcv/unhcv/projects/diffusion/inpainting/evaluation/evaluation_metric.py \
--result_path ${result_root}/result \
--data_indexes_path $data_indexes_path \
--show_path ${result_root}/evaluation \
--show --inter_ratio_thres 0.95

