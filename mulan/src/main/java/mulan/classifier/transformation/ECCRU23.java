package mulan.classifier.transformation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Random;

import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelOutput;
import mulan.data.ImbalancedStatistics;
import mulan.data.MultiLabelInstances;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.RemovePercentage;

/**
 * <p>Implementation of the ECCRU2 and ECCRU3 algorithm.</p> 
 * <p>For more information, see <em> Liu, Bin Tsoumakas, Grigorios. 
 * "Making Classifier Chains Resilient to Class Imbalance." ACML. 2018. pp.280-295.</em></p>
 *
 * @author Bin Liu
 * @version 2018.5.10
 */

public class ECCRU23 extends ECCRU {
    protected int maxNumofModels;
    protected double theta_min=0.5;
	protected double theta_max=10;
	protected int numAcutualModels;
    protected int numModelsPerLabel[]; //i-th element is the number of models built for i-th label
    protected boolean isPlusVersion=false;  // if True, use theta_min to decide the minimal number of classifiers built for each label (ECCRU3), otherwise ECCRU2  
    protected int seed=1;

	protected ArrayList<HashMap<Integer,Integer>> labelIndexMapList; 
	protected ArrayList<Remove> removeList;
	protected ArrayList<Integer> removeLabelIndexList;  //the label index removed from the mlTrainSet
	protected ArrayList<Integer> remainLabelIndexList;  //the label index remained in mlTrainSet
	/** 
    * Default constructor
     */
    public ECCRU23() {
        this(new J48(), 10, true, true);
    }

    /**
     * Creates a new object
     *
     * @param classifier the base classifier for each ClassifierChain model
     * @param aNumOfModels the number of models
     * @param doUseConfidences whether to use confidences or not
     * @param doUseSamplingWithReplacement whether to use sampling with replacement or not 
     */
    public ECCRU23(Classifier classifier, int aNumOfModels,
            boolean doUseConfidences, boolean doUseSamplingWithReplacement) {
    	super(classifier,aNumOfModels,doUseConfidences,doUseSamplingWithReplacement);
    }
    
  	/**
  	 * @return the isPlusVersion
  	 */
  	public boolean isPlusVersion() {
  		return isPlusVersion;
  	}

  	/**
  	 * @param isPlusVersion the isPlusVersion to set
  	 */
  	public void setPlusVersion(boolean isPlusVersion) {
  		this.isPlusVersion = isPlusVersion;
  	}
    
  	 /**
	 * @return the theta_min
	 */
	public double getTheta_min() {
		return theta_min;
	}

	/**
	 * @param theta_min the theta_min to set
	 */
	public void setTheta_min(double theta_min) {
		this.theta_min = theta_min;
	}
    
    /**
	 * @return the theta_max
	 */
	public double getTheta_max() {
		return theta_max;
	}

	/**
	 * @param theta_max the theta_max to set
	 */
	public void setTheta_max(double theta_max) {
		this.theta_max = theta_max;
	}
	
	protected void initializeVariable(MultiLabelInstances train){
		maxNumofModels=(int)(numOfModels*theta_max);
		ensemble = new ClassifierChain[maxNumofModels];
		
		ImbalancedStatistics is =new ImbalancedStatistics();
		is.calculateImSta(train);
		int c1[]=is.getC1();
		int c0[]=is.getC0();
		
		numModelsPerLabel=new int [numLabels];
		double a=0;
		int c=0;
		for(int i=0;i<c1.length;i++){
			c1[i]=Math.min(c1[i], c0[i]);  //number of minority class instances
			if(c1[i]!=0){
				a+=c1[i];
				c++;
			}
		}
		a/=c;
		for(int i=0;i<c1.length;i++){
			if(c1[i]!=0){
				numModelsPerLabel[i]=(int)(numOfModels*a/c1[i]);
				if(isPlusVersion){
					numModelsPerLabel[i]=Math.max(numModelsPerLabel[i], (int)(numOfModels*theta_min));
				}
				else{
					numModelsPerLabel[i]=Math.max(numModelsPerLabel[i], 1);
				}
				numModelsPerLabel[i]=Math.min(numModelsPerLabel[i],maxNumofModels);
			}
			else{
				numModelsPerLabel[i]=numOfModels;
			}
			
			
		}
		
		removeLabelIndexList=new ArrayList<>(numLabels);
		remainLabelIndexList=new ArrayList<>(numLabels);
		for(int i=0;i<numLabels;i++){
			remainLabelIndexList.add(i);
		}
		labelIndexMapList=new ArrayList<>();
		removeList=new ArrayList<>();	
	}
      
    @Override
    protected void buildInternal(MultiLabelInstances trainingSet) throws Exception {	
    	initializeVariable(trainingSet);
    	
        Instances dataSet = new Instances(trainingSet.getDataSet());
        Random rand=new Random(seed);
        
        MultiLabelInstances processedTrainingSet=null;
        Instances sampledDataSet=null;
        
        for (int i = 0; i < maxNumofModels; i++) {
        	if(remainLabelIndexList.size()<=1){
        		break;  
        	}
    		numAcutualModels=i+1;
            dataSet.randomize(new Random(rand.nextInt()));
            if (useSamplingWithReplacement) {
                int bagSize = dataSet.numInstances() * BagSizePercent / 100;
                // create the in-bag dataset
                sampledDataSet = dataSet.resampleWithWeights(new Random(1));
                if (bagSize < dataSet.numInstances()) {
                    sampledDataSet = new Instances(sampledDataSet, 0, bagSize);
                }
                processedTrainingSet =new MultiLabelInstances(sampledDataSet,trainingSet.getLabelsMetaData());
            } else {
                RemovePercentage rmvp = new RemovePercentage();
                rmvp.setInvertSelection(true);
                rmvp.setPercentage(samplingPercentage);
                rmvp.setInputFormat(dataSet);
                sampledDataSet = Filter.useFilter(dataSet, rmvp);
                processedTrainingSet =new MultiLabelInstances(sampledDataSet,trainingSet.getLabelsMetaData());
            }    

            //construct new training set and record the label index correspondence of original and new training set
            if(removeLabelIndexList.size()>0){
            	int removeArray[]=new int [removeLabelIndexList.size()];
            	for(int j=0;j<removeArray.length;j++){
            		removeArray[j]=labelIndices[removeLabelIndexList.get(j)];
            	}
            	Remove remove=new Remove();
            	remove.setAttributeIndicesArray(removeArray);
            	remove.setInputFormat(sampledDataSet);
            	remove.setInvertSelection(false);
				Instances filteredTrainIns=Filter.useFilter(sampledDataSet, remove);
				processedTrainingSet=processedTrainingSet.reintegrateModifiedDataSet(filteredTrainIns);
				
				removeList.add(remove);
				
				HashMap map=new HashMap<Integer,Integer>(remainLabelIndexList.size());
				for(int newIndex=0;newIndex<remainLabelIndexList.size();newIndex++){
					map.put(newIndex, remainLabelIndexList.get(newIndex));
				}
				labelIndexMapList.add(map);
            }
            else{
            	removeList.add(null);
            	labelIndexMapList.add(null);
            }
            
            int[] chain = new int[remainLabelIndexList.size()]; //remaining number of labels
            for (int j = 0; j < remainLabelIndexList.size(); j++) {
                chain[j] = j;
            }
            for (int j = 0; j < chain.length; j++) {                
            	int randomPosition = rand.nextInt(chain.length);
            	
                int temp = chain[j];
                chain[j] = chain[randomPosition];
                chain[randomPosition] = temp;
            }
            ensemble[i] = new ClassifierChainUnderSampling(baseClassifier, chain,underSamplingPercent);
           
            ensemble[i].build(processedTrainingSet); 

            
            //update numModelsPerLabel, remainLabelIndexList and removeLabelIndexList
            for(int j=0;j<numModelsPerLabel.length;j++){
            	if(numModelsPerLabel[j]!=0){
            		numModelsPerLabel[j]--;
            	}
            }
            Iterator<Integer> it=remainLabelIndexList.iterator();
            for(int j=0;j<remainLabelIndexList.size();j++){
            	int index=it.next();
            	if(numModelsPerLabel[index]==0){
            		removeLabelIndexList.add(index);
            		it.remove();
            		j--;            		
            	}
            }
        }
        
       
        if(measure!=thresholdOptimizationMeasures.None){
        	calculateThresholds(trainingSet);         	
        }
    }
    
    @Override
    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception, InvalidDataException {

        int[] sumVotes = new int[numLabels];
        double[] sumConf = new double[numLabels];
        int[] counts=new int[numLabels];
        Arrays.fill(sumVotes, 0);
        Arrays.fill(sumConf, 0);
        Arrays.fill(counts, 0);

        for (int i = 0; i < numAcutualModels; i++) {
        	MultiLabelOutput ensembleMLO;
        	Remove remove=removeList.get(i);
        	HashMap<Integer,Integer> map=labelIndexMapList.get(i);
        	if(remove!=null){
        		remove.input(instance);
                remove.batchFinished();
                Instance processedIns=remove.output();
                ensembleMLO = ensemble[i].makePrediction(processedIns);
        	}
        	else{
        		ensembleMLO = ensemble[i].makePrediction(instance);
        	}

            boolean[] bip = ensembleMLO.getBipartition();
            double[] conf = ensembleMLO.getConfidences();
            
            if(map!=null){
            	for (Map.Entry<Integer, Integer> entry : map.entrySet()){
            		sumVotes[entry.getValue()] += bip[entry.getKey()] == true ? 1 : 0;
                    sumConf[entry.getValue()] += conf[entry.getKey()];
                    counts[entry.getValue()]++;
            	}
            }
            else{
            	for (int j = 0; j < numLabels; j++) {
                	sumVotes[j] += bip[j] == true ? 1 : 0;
                    sumConf[j] += conf[j];
                    counts[j]++;
                }
            }
        }

        double[] confidence = new double[numLabels];
        for (int j = 0; j < numLabels; j++) {
            if (useConfidences) {
                confidence[j] = sumConf[j] / counts[j];
            } else {
                confidence[j] = sumVotes[j] / (double) counts[j];
            }
        }
        
        MultiLabelOutput mlo;
        if(measure==thresholdOptimizationMeasures.None){
        	mlo = new MultiLabelOutput(confidence, 0.5);
        }
        else{
        	boolean bipartition[]=new boolean[numLabels];
        	for(int j=0;j<numLabels;j++){
        		bipartition[j]=confidence[j]>=thresholds[j];
        	}
        	mlo=new MultiLabelOutput(bipartition, confidence);
        }
        
        return mlo;
    }

    
}
