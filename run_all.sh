#CAMRa2011 dataset
python main.py --dataset=CAMRa2011 --batch_size=256 --num_negatives=4 --beta=0.025 --threshold=3 --epoch=30 > CAMRa2011_optimal.log   

#MovieLens dataset
python main.py --dataset=MovieLens --batch_size=128 --beta=0.025 --threshold=2 --epoch=50 > MovieLens_optimal.log 


#Mafenwo dataset
python main.py --dataset=Mafengwo --num_negatives=10 --beta=0.1 --epoch=300 --lr=0.0001 > Mafengwo_optimal.log   

#MafengwoS dataset
python main.py --dataset=MafengwoS --num_negatives=18 --beta=0.025 --threshold=3 --epoch=300 --lr=0.0001 > MafengwoS_optimal.log


 
