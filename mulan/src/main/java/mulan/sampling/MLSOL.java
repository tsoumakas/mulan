package mulan.sampling;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import mulan.data.ImbalancedStatistics;
import mulan.data.MultiLabelInstances;
import weka.core.DenseInstance;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.KnnResult;
import weka.core.neighboursearch.LinearNNSearch2;
/**
 * 
 *
 * Implementation of MLSOL.  
 * 
 * 
 * @author Bin Liu
 * @version 2019.6.10
 *
 */


public class MLSOL extends MultiLabelSampling {
	protected double weights[];
	protected Double C [][];   //the C_ij for majority class is null
	protected InstanceType insTypes[][];
	protected int knnIndices[][];
	protected String minLabels[];
	protected int labelIndices[];
	protected int featureIndices[];
	
	protected double sumW;
	protected int numOfNeighbors=5;
	protected LinearNNSearch2 lnn = new LinearNNSearch2();
	protected DistanceFunction dfunc = new EuclideanDistance();
	
	public InstanceType[][] getInsType(){
		return insTypes;
	}
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
		int generatingNumIns=(int)(P*oriNumIns);
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
		initilizeInsTypes(ins);
		Instances insNew=new Instances(ins);
        		
		for(int i=0;i<generatingNumIns;i++){
			double d=rnd.nextDouble()*sumW;
			int centralIndex=-1;
			double s=0;
			for(int j=0;j<weights.length;j++){
				s+=weights[j];
				if(d<=s){
					centralIndex=j;
					break;
				}
			}
			int referenceIndex=knnIndices[centralIndex][rnd.nextInt(numOfNeighbors)];
			Instance newData=generateSyntheticInstance(ins.get(centralIndex), ins.get(referenceIndex), centralIndex, referenceIndex, rnd);
			insNew.add(newData);
		}
		
		
		return new MultiLabelInstances(insNew, mlDataset.getLabelsMetaData());
	}
	
	protected void calculateWeight(Instances ins) throws Exception{
		int numIns=ins.numInstances();
		int numLabels=labelIndices.length;
		
		String labelIndicesString = "";
        for (int i = 0; i < numLabels-1; i++) {
            labelIndicesString += (labelIndices[i] + 1) + ",";
        }
        labelIndicesString += (labelIndices[numLabels-1]+1); 
        dfunc.setAttributeIndices(labelIndicesString);
        dfunc.setInvertSelection(true);
		
        lnn.setDistanceFunction(dfunc);
        lnn.setInstances(ins);
        lnn.setMeasurePerformance(false);
		
        for(int i=0;i<numIns;i++){
			Instance data=ins.get(i);
			KnnResult result=lnn.kNearestNeighbours2(data, numOfNeighbors); // the number of instance in knn may larger than numOfNeighbors
			for(int k=0;k<numOfNeighbors;k++){
				knnIndices[i][k]=result.indices[k];
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
        
        sumW=0;
        for(int i=0;i<numIns;i++){
        	weights[i]=0;
        	for(int j=0;j<numLabels;j++){
        		if(scores[i][j]!=null){
        			weights[i]+=scores[i][j];
        		}
        	}
        	sumW+=weights[i];
        }
	}

	protected void initilizeInsTypes(Instances ins){
		int numIns=ins.numInstances();
		insTypes=new InstanceType[numIns][labelIndices.length];
		
		for(int i=0;i<numIns;i++){
			Instance data=ins.get(i);
			for(int j=0;j<labelIndices.length;j++){
				if(data.stringValue(labelIndices[j]).equals(minLabels[j])){
					if(C[i][j]<0.3){
						insTypes[i][j]=InstanceType.SAFE;
					}
					else if(C[i][j]<0.7){
						insTypes[i][j]=InstanceType.BORDERLINE;
					}
					else if(C[i][j]<1 ){   
						insTypes[i][j]=InstanceType.RARE;
					}
					else{
						insTypes[i][j]=InstanceType.OUTLIER;
					}
				}
				else{
					insTypes[i][j]=InstanceType.MAJORITY;
				}
			}
		}
		//re-analyse the RARE type
        boolean flag=false;
        do{
        	flag=false; 
        	for(int i=0;i<numIns;i++){
             	for(int j=0;j<labelIndices.length;j++){
             		if(insTypes[i][j]==InstanceType.RARE){
             			for(int k:knnIndices[i]){
             				if(insTypes[k][j]==InstanceType.SAFE ||insTypes[i][j]==InstanceType.BORDERLINE){
             					insTypes[i][j]=InstanceType.BORDERLINE;
             					flag=true;
             					break;
             				}
             			}
             		}
             	}
        	}
        }while(flag);
	}
	
	protected Instance generateSyntheticInstance(Instance centralInstance,Instance referenceInstance,int centralIndex,int referenceIndex,Random rnd){
		Instance synthetic =new DenseInstance(centralInstance);
		for(int i=0;i<featureIndices.length;i++){
			int j=featureIndices[i];
			if(centralInstance.attribute(j).isNumeric()||centralInstance.attribute(j).isDate()){
				synthetic.setValue(j, centralInstance.value(j)+rnd.nextDouble()*(referenceInstance.value(j)-centralInstance.value(j)));
			}
			else{				
				synthetic.setValue(j,rnd.nextBoolean()?centralInstance.value(j):referenceInstance.value(j));
			}
		}
		double d1=dfunc.distance(centralInstance, synthetic);
		double d2=dfunc.distance(referenceInstance, synthetic);
		double cd=d1/(d1+d2);
		if(Double.isNaN(cd)){  //d1=d2=0
			cd=0.5; // the distances with two instances are same
		}
		double theta=0.5;
		
		for(int i=0;i<labelIndices.length;i++){			
			int j=labelIndices[i];
			if(centralInstance.stringValue(j).equals(referenceInstance.stringValue(j))){
				synthetic.setValue(j,centralInstance.value(j));
			}
			else{
				if(insTypes[centralIndex][i]==InstanceType.MAJORITY){
					//swap centralInstance and referenceInstance
					Instance tInstance=centralInstance; centralInstance=referenceInstance; referenceInstance=tInstance;
					int t=centralIndex; centralIndex=referenceIndex; referenceIndex=t;
					cd=1.0-cd;
				}
				switch (insTypes[centralIndex][i]){
					case SAFE:
						theta=0.5;	break;
					case BORDERLINE:
						theta=0.75;	break;
					case RARE:
						theta=1.0+1e-5;	break;
					case OUTLIER:
						theta=0.0-1e-5;	break;
				}
				if(cd<=theta){
					synthetic.setValue(j,centralInstance.value(j));
				}
				else{
					synthetic.setValue(j,referenceInstance.value(j));
				}
			}
		}
		return synthetic;
	}
}