package mulan.sampling;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.Random;

import mulan.data.ImbalancedStatistics;
import mulan.data.MultiLabelInstances;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.KnnResult;
import weka.core.neighboursearch.LinearNNSearch2;

/**
 * 
 *
 * Implementation of MLUL.  
 * 
 * Correction Version for TKDE submission </p> 
 * 
 *
 * @author Bin Liu
 * @version 2019.10.17
 *
 */

public class MLUL extends MultiLabelSampling {

	protected double weights[];
	protected Double C [][];   //the C_ij for majority class is null
	protected int knnIndices[][];
	protected String minLabels[];
	protected int labelIndices[];
	protected int featureIndices[];
	
	protected double sumW;
	protected int numOfNeighbors=5;
	protected LinearNNSearch2 lnn = new LinearNNSearch2();
	protected DistanceFunction dfunc = new EuclideanDistance();
	
	protected ArrayList<ArrayList<Integer>> rknnIndicesList; //rknnIndicesList.get(i) store the indices of reverse kNNs of i-th instance

	
	public Double[][] getC(){
		return C;
	}
	/** 
	 * @return the numOfNeighbors
	 */
	public int getNumOfNeighbors() {
		return numOfNeighbors;
	}
	/**
	 * @param numOfNeighbors the numOfNeighbors to set
	 */
	public void setNumOfNeighbors(int numOfNeighbors) {
		this.numOfNeighbors = numOfNeighbors;
	}
		
	
	
	@Override
	public MultiLabelInstances build(MultiLabelInstances mlDataset) throws Exception {
		Random rnd=new Random(seed);
		int numLabels=mlDataset.getNumLabels();
		labelIndices=mlDataset.getLabelIndices();
		featureIndices=mlDataset.getFeatureIndices();
		int oriNumIns=mlDataset.getNumInstances();
		int deleteNumIns=(int)(P*oriNumIns);
		weights=new double[oriNumIns];
		minLabels=new String[numLabels];
		
		C=new Double[oriNumIns][numLabels];
		knnIndices=new int[oriNumIns][numOfNeighbors];
		
		ImbalancedStatistics is=new ImbalancedStatistics();
		is.calculateC0C1(mlDataset);		
		int c1[]=is.getC1();
		int c0[]=is.getC0();
		for(int j=0;j<numLabels;j++){
			if(c1[j]>c0[j]){
				minLabels[j]="0";
			}
			else{
				minLabels[j]="1";
			}
		}

		Instances ins=mlDataset.getDataSet();
		calculateWeight(ins);
		
		Instances insNew=new Instances(ins);
		
		ArrayList<Integer> deleteIndices = selectDeleteIndicesRandomly(insNew.numInstances(), deleteNumIns, rnd); 						
		for(int i:deleteIndices){
			insNew.remove(i);
		}
		

		
		MultiLabelInstances mlDatasetNew=new MultiLabelInstances(insNew, mlDataset.getLabelsMetaData());
		return mlDatasetNew;
	}
	
	protected void calculateWeight(Instances ins) throws Exception{
		int numIns=ins.numInstances();
		int numLabels=labelIndices.length;
		
		String labelIndicesString = "";
        for (int i = 0; i < numLabels-1; i++) {
            labelIndicesString += (labelIndices[i] + 1) + ",";
        }
        labelIndicesString += (labelIndices[numLabels-1]+1);  //labelIndicesString += labelIndices[numLabels-1]; 
        dfunc.setAttributeIndices(labelIndicesString);
        dfunc.setInvertSelection(true);
		
        lnn.setDistanceFunction(dfunc);
        lnn.setInstances(ins);
        lnn.setMeasurePerformance(false);
		
        //initialize rknnIndicesList
        rknnIndicesList=new ArrayList<ArrayList<Integer>>(numIns);
        for(int i=0;i<numIns;i++){
        	ArrayList<Integer> list=new ArrayList<Integer>();
        	rknnIndicesList.add(list);
        }
        
        for(int i=0;i<numIns;i++){
			Instance data=ins.get(i);
			KnnResult result=lnn.kNearestNeighbours2(data, numOfNeighbors); // the number of instance in knn may larger than numOfNeighbors
			for(int k=0;k<numOfNeighbors;k++){
				knnIndices[i][k]=result.indices[k];
				rknnIndicesList.get(knnIndices[i][k]).add(i);
			}			
			
			for(int j=0;j<numLabels;j++){
				int numMaj=0;
				if(data.stringValue(labelIndices[j]).equals(minLabels[j])){
					for(int k=0;k<numOfNeighbors;k++){
						if(!data.stringValue(labelIndices[j]).equals(result.knn.get(k).stringValue(labelIndices[j]))){
							numMaj++;
						}
					}
					C[i][j]=numMaj*1.0/numOfNeighbors;  //Eqation (1)
				}
				else{ //majority class
					C[i][j]=null; 
				}
			}
		}
        
                
        //Transform the C to scores
		Double scores[][]=new Double[numIns][numLabels];
        for(int j=0;j<numLabels;j++){
        	double sum=0;
        	int c=0;
        	for(int i=0;i<numIns;i++){
        		if(C[i][j]!=null && C[i][j]<1){
        			sum+=C[i][j];
        			c++;
        		}
        	}
        	if(c!=0 && sum!=0.0){ // add condition sum!=0.0
        		for(int i=0;i<numIns;i++){
        			if(C[i][j]!=null && C[i][j]<1){
        				scores[i][j]=C[i][j]/sum;
        			}
        		}
        	}
        }
        
        //calculate w
        double weights1[]=new double[numIns];
        double sumW1=0;
        for(int i=0;i<numIns;i++){
        	weights1[i]=0;
        	for(int j=0;j<numLabels;j++){
        		if(scores[i][j]!=null){  //y_ij is minority class and y_ij is not an outlier
        			weights1[i]+=scores[i][j];
        		}
        	}
        	sumW1+=weights1[i];
        }
        
        //calculate v
        sumW=0;
        double minW=Double.MAX_VALUE;
        for(int i=0;i<numIns;i++){
        	double v1=0; //the first part of equation for calculating v
        	for(int rknnIndex:rknnIndicesList.get(i)){
        		for(int j=0;j<numLabels;j++){
					if(scores[rknnIndex][j]!=null){ //y_rj is minority class and y_rj is not an outlier
						if(scores[i][j]==null){  //y_ij is majority class or y_ij is an outlier
							v1+=scores[rknnIndex][j];
		        		}
						else{  //y_ij is minority class and y_ij is not an outlier
							v1-=scores[rknnIndex][j];
						}
					}
				}
			}
        	
        	if(rknnIndicesList.get(i).size()!=0){
    			v1/=rknnIndicesList.get(i).size();
    		}
        	weights[i]=weights1[i]-v1;
        	
        	if(minW>weights[i]){
        		minW=weights[i];
        	}
        }
  
        //transform all weights values to positive or zero
        if(minW<0){
         	for(int i=0;i<numIns;i++){
             	weights[i]-=minW;
             }
         }
       
        
        sumW=0;
        for(int i=0;i<numIns;i++){
        	sumW+=weights[i];
        }
	}

	protected ArrayList<Integer> selectDeleteIndicesRandomly(int currentInsNum,int currentDeleteInsNum,Random rnd){
		int currentRetainInsNum=currentInsNum-currentDeleteInsNum;
		HashSet<Integer> retainInsIndexSet=new HashSet<>(currentRetainInsNum);
		ArrayList<Integer> oWeightsIndexlist=new ArrayList<>();
		for(int i=0;i<currentInsNum;i++){
			if(!Double.isNaN(weights[i])&&weights[i]<=0){ //the weight[i]=NaN if i is not in the retainInsIndexSet 
				oWeightsIndexlist.add(i);
			}
		}
		//number of zero weight is less than number of deleted instance in this round
		if(oWeightsIndexlist.size()<currentDeleteInsNum){   
			retainInsIndexSet.clear();
			while(retainInsIndexSet.size()<currentRetainInsNum){
				double d=rnd.nextDouble()*sumW;
				double s=0;
				for(int i=0;i<currentInsNum;i++){
					if(!Double.isNaN(weights[i])){
						s+=weights[i];
						if(d<=s){
							retainInsIndexSet.add(i);
							sumW-=weights[i];
							weights[i]=0.0;
							break;
						}
					}
				}
			}
		}
		//number of zero weight is equal to or large than number of deleted instance in this round
		else{
			int indices[]=randomIntArray(oWeightsIndexlist.size()-1, 0, currentDeleteInsNum,rnd);
			retainInsIndexSet.clear();
			for(int i=0;i<currentInsNum;i++){
				retainInsIndexSet.add(i);
			}
			for(int i:indices){
				retainInsIndexSet.remove(oWeightsIndexlist.get(i));
			}
		}
		
		ArrayList<Integer> deleteIndices=new ArrayList<>(currentDeleteInsNum);
		for(int i=0;i<currentInsNum;i++){
			if(!retainInsIndexSet.contains(i)){
				deleteIndices.add(i);
			}
		}
		Collections.sort(deleteIndices); //ascending order
		Collections.reverse(deleteIndices); //descending order
		return deleteIndices;
	}

	 /** Returns an array of random number without replicated in a certain range
	 * 
	 * @param min the minimum
	 * @param max the maximum
	 * @param num the count of numbers that returns
	 * @return an array of random number in the range of [min,max]
	 */
	protected int[] randomIntArray(int max,int min,int num, Random rnd){
		int n[]=new int[num];
		int lenght=max-min+1;
		int tag[]=new int[lenght];
		
		for(int i=0;i<lenght;i++){
			tag[i]=0; 
		}
		int r;
		for(int i=0;i<num;i++){
			do{
				r=Math.abs(rnd.nextInt())%(max-min+1)+min;
			}while(tag[r-min]==1);
			n[i]=r;
			tag[r-min]=1;
		}
		return n;
	}


}
