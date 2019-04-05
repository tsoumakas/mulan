package mulan.sampling;

import java.util.ArrayList;
import java.util.Random;

import mulan.core.Util;
import mulan.data.ImbalancedStatistics;
import mulan.data.MultiLabelInstances;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.LinearNNSearch;

/**
 * <p>Implementation of the MLSMOTE.</p> <p>For
 * more information, see <em> Charte, Francisco, et al. 
 * "MLSMOTE: Approaching imbalanced multilabel learning through synthetic instance generation."
 *  Knowledge-Based Systems 89 (2015): 385-397.</em></p>
 *
 * @author Bin Liu
 * @version 2019.3.19
 *
 */

public class MLSMOTE extends MultiLabelSampling{
	public static void main(String[] args) {
		 
		try{
			String dataName="bibtex";
			
			String path = "F://刘彬学校电脑资料//希腊//数据//MutliLabel Datasets//"+dataName+"//";;
			String trainDatasetPath = path +dataName +".arff";
		    String xmlLabelsDefFilePath = path + dataName +".xml";
		    MultiLabelInstances mlTrainData = new MultiLabelInstances(trainDatasetPath,xmlLabelsDefFilePath);
		   
		    MLSMOTE mlsmote=new MLSMOTE();
		    MultiLabelInstances newMlTrainData=mlsmote.build(mlTrainData);
		    
		    ImbalancedStatistics is=new ImbalancedStatistics();
			is.calculateImSta(mlTrainData);
			System.out.println("ORI "+mlTrainData.getNumInstances() +"\n"+is.toString());
			is.calculateImSta(newMlTrainData);
			System.out.println("SAMPLING "+newMlTrainData.getNumInstances() +"\n"+is.toString());
		    
		    
		}
		catch (Exception e){
			e.printStackTrace();
		}
		 
	}
	
	public enum LabelGeneration {
        Intersection, Union, Ranking,
    };
	
	
	private int numOfNeighbors=5;
	private LinearNNSearch lnn = new LinearNNSearch();
	private DistanceFunction dfunc = new EuclideanDistance();  
	private LabelGeneration labelGenerationMethod=LabelGeneration.Ranking;
	
	private int labelIndices[];
	private int featureIndices[];
	
	/**
	 * @return the numOfNeighbors
	 */
	public int getNumOfNeighbors() {
		return numOfNeighbors;
	}

	/**
	 * @return the lnn
	 */
	public LinearNNSearch getLnn() {
		return lnn;
	}

	/**
	 * @return the dfunc
	 */
	public DistanceFunction getDfunc() {
		return dfunc;
	}

	/**
	 * @return the labelGenerationMethod
	 */
	public LabelGeneration getLabelGenerationMethod() {
		return labelGenerationMethod;
	}

	/**
	 * @param numOfNeighbors the numOfNeighbors to set
	 */
	public void setNumOfNeighbors(int numOfNeighbors) {
		this.numOfNeighbors = numOfNeighbors;
	}

	/**
	 * @param lnn the lnn to set
	 */
	public void setLnn(LinearNNSearch lnn) {
		this.lnn = lnn;
	}

	/**
	 * @param dfunc the dfunc to set
	 */
	public void setDfunc(DistanceFunction dfunc) {
		this.dfunc = dfunc;
	}

	/**
	 * @param labelGenerationMethod the labelGenerationMethod to set
	 */
	public void setLabelGenerationMethod(LabelGeneration labelGenerationMethod) {
		this.labelGenerationMethod = labelGenerationMethod;
	}


	public MLSMOTE(int numOfNeighbors){
		super();
		this.numOfNeighbors=numOfNeighbors;
	}
	
	public MLSMOTE(){
		super();
	}
	
	@Override
	public MultiLabelInstances build(MultiLabelInstances mlDataset) throws Exception {
		
		Random rnd=new Random(seed);
		Instances ins=mlDataset.getDataSet();

		int numLabels=mlDataset.getNumLabels();
		labelIndices=mlDataset.getLabelIndices();
		featureIndices=mlDataset.getFeatureIndices();

		
		String labelIndicesString = "";
        for (int i = 0; i < numLabels - 1; i++) {
            labelIndicesString += (labelIndices[i] + 1) + ",";
        }
        labelIndicesString += (labelIndices[numLabels - 1] + 1);
        dfunc.setAttributeIndices(labelIndicesString);
        dfunc.setInvertSelection(true);
		
		
		ImbalancedStatistics is=new ImbalancedStatistics();
		is.calculateImSta(mlDataset);
		double meanIR=is.getMeanIR();
		double IRLbls[]=is.getIRLbls();
		ArrayList<Integer> minLabelIndexList=new ArrayList<>();
		for(int i=0;i<numLabels;i++){
			if(IRLbls[i]>meanIR){
				minLabelIndexList.add(i);
			}
		}
		

        Instances minBag=new Instances(ins,0);
        MultiLabelInstances newMlDataset=new MultiLabelInstances(new Instances(ins), mlDataset.getLabelsMetaData());
        for(int j=0;j<labelIndices.length;j++){
        	if(IRLbls[j]<meanIR){
        		continue;
        	}
        	
        	int labelIndex=labelIndices[j];
        	minBag.clear();
        	Instances ins2=newMlDataset.getDataSet();
        	for(int i=0;i<ins2.numInstances();i++){
        		if(ins2.get(i).stringValue(labelIndex).equals("1")){
        			minBag.add(ins2.get(i));
        		}
        	}
        	
        	if(minBag.size()<=1){
    			continue;
    		}       	
        	

            lnn.setDistanceFunction(dfunc);
            lnn.setInstances(minBag);
            lnn.setMeasurePerformance(false);
        	
        	for(Instance data:minBag){
        		Instances knn = lnn.kNearestNeighbours(data, numOfNeighbors);
        		int insIndex=Util.randomInt(0, knn.size()-1, rnd);
        		Instance newData=generateSyntheticInstance(data, insIndex, knn, rnd);
        		newMlDataset.getDataSet().add(newData);
        	}
        }

		return newMlDataset;
	}
	
	private Instance generateSyntheticInstance(Instance centralInstance,int refNeighbourIndex,Instances knn,Random rnd){
		Instance synthetic =new DenseInstance(centralInstance);
		synthetic.setDataset(knn);
		for(int i:featureIndices){
			if(centralInstance.attribute(i).isNumeric()||centralInstance.attribute(i).isDate()){
				synthetic.setValue(i, centralInstance.value(i)+rnd.nextDouble()*(knn.get(refNeighbourIndex).value(i)-centralInstance.value(i)));
			}
			else{
				Attribute attr=centralInstance.attribute(i);
				int[] valueCounts = new int[attr.numValues()];
				
				int iVal = (int)centralInstance.value(attr);
				valueCounts[iVal]++;
				for(Instance neighbour:knn){
					iVal = (int)neighbour.value(attr);
					valueCounts[iVal]++;
				}
				
				int maxIndex = 0;
			    int max = Integer.MIN_VALUE;
			    for (int idx = 0; idx < attr.numValues(); idx++) {
			    	if (valueCounts[idx] > max){
			          max = valueCounts[idx];
			          maxIndex = idx;
			        }
			    }
			    synthetic.setValue(i, attr.value(maxIndex));
			}
		}
		
		//String newLabels[]=new String[labelIndices.length];
		int c0,c1;
		switch (labelGenerationMethod) {
		case Ranking:
			for(int i:labelIndices){
				c0=c1=0;
				if(centralInstance.stringValue(i).equals("1"))
					c1++;
				else if(centralInstance.stringValue(i).equals("0"))
					c0++;
				for(Instance neighbour:knn){
					if(neighbour.stringValue(i).equals("1"))
						c1++;
					else if(neighbour.stringValue(i).equals("0"))
						c0++;
				}
				if(c1*2>=(c0+c1)){
					synthetic.setValue(i, "1");
				}
				else{
					synthetic.setValue(i, "0");
				}
			}
			break;
		
		case Union:
			for(int i:labelIndices){
				c0=c1=0;
				if(centralInstance.stringValue(i).equals("1"))
					c1++;
				else if(centralInstance.stringValue(i).equals("0"))
					c0++;
				for(Instance neighbour:knn){
					if(neighbour.stringValue(i).equals("1"))
						c1++;
					else if(neighbour.stringValue(i).equals("0"))
						c0++;
				}
				if(c1>0){
					synthetic.setValue(i, "1");
				}
				else{
					synthetic.setValue(i, "0");
				}
			}
			break;
		

		case Intersection:
			for(int i:labelIndices){
				c0=c1=0;
				if(centralInstance.stringValue(i).equals("1"))
					c1++;
				else if(centralInstance.stringValue(i).equals("0"))
					c0++;
				for(Instance neighbour:knn){
					if(neighbour.stringValue(i).equals("1"))
						c1++;
					else if(neighbour.stringValue(i).equals("0"))
						c0++;
				}
				if(c1==knn.size()+1){
					synthetic.setValue(i, "1");
				}
				else{
					synthetic.setValue(i, "0");
				}
			}
			
			break;
		}

		return synthetic;
	}
	

}
