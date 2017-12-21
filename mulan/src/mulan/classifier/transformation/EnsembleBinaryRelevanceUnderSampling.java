package mulan.classifier.transformation;

import java.util.Arrays;
import java.util.Random;

import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.measure.InformationRetrievalMeasures;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

public class EnsembleBinaryRelevanceUnderSampling extends TransformationBasedMultiLabelLearner{

	protected int seed=1;
	protected int numOfModels=10;
	protected BinaryRelevanceUnderSampling[] ensemble;
	protected double underSamplingPercent=1.0;
	protected boolean useFmeasureOptimizationThreshold=false;
	protected boolean useConfidences=true;
	double thresholds[];
	
	public EnsembleBinaryRelevanceUnderSampling(){
		super();
	}
	
	public EnsembleBinaryRelevanceUnderSampling(Classifier baseClassifier){
		super(baseClassifier);
	}
	
	
	
    /**
	 * @return the seed
	 */
	public int getSeed() {
		return seed;
	}

	/**
	 * @return the numOfModels
	 */
	public int getNumOfModels() {
		return numOfModels;
	}

	/**
	 * @return the underSamplingPercent
	 */
	public double getUnderSamplingPercent() {
		return underSamplingPercent;
	}

	/**
	 * @return the useFmeasureOptimizationThreshold
	 */
	public boolean isUseFmeasureOptimizationThreshold() {
		return useFmeasureOptimizationThreshold;
	}

	/**
	 * @return the useConfidences
	 */
	public boolean isUseConfidences() {
		return useConfidences;
	}

	/**
	 * @param seed the seed to set
	 */
	public void setSeed(int seed) {
		this.seed = seed;
	}

	/**
	 * @param numOfModels the numOfModels to set
	 */
	public void setNumOfModels(int numOfModels) {
		this.numOfModels = numOfModels;
	}

	/**
	 * @param underSamplingPercent the underSamplingPercent to set
	 */
	public void setUnderSamplingPercent(double underSamplingPercent) {
		this.underSamplingPercent = underSamplingPercent;
	}

	/**
	 * @param useFmeasureOptimizationThreshold the useFmeasureOptimizationThreshold to set
	 */
	public void setUseFmeasureOptimizationThreshold(boolean useFmeasureOptimizationThreshold) {
		this.useFmeasureOptimizationThreshold = useFmeasureOptimizationThreshold;
	}

	/**
	 * @param useConfidences the useConfidences to set
	 */
	public void setUseConfidences(boolean useConfidences) {
		this.useConfidences = useConfidences;
	}


	
	
	@Override
	protected void buildInternal(MultiLabelInstances trainingSet) throws Exception {
		
		Random rand=new Random(seed);
		ensemble=new BinaryRelevanceUnderSampling[numOfModels];
	    for (int i = 0; i < numOfModels; i++) {
	        debug("EBRUS Building Model:" + (i + 1) + "/" + numOfModels);
	        ensemble[i]=new BinaryRelevanceUnderSampling(baseClassifier,underSamplingPercent,rand.nextInt());
	        ensemble[i].build(trainingSet); 
	    }
	    
	    if(useFmeasureOptimizationThreshold){
	    	calculateThresholds(trainingSet);
	    }
	}

	@Override
	protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception, InvalidDataException {
        int[] sumVotes = new int[numLabels];
        double[] sumConf = new double[numLabels];

        Arrays.fill(sumVotes, 0);
        Arrays.fill(sumConf, 0);

        for (int i = 0; i < numOfModels; i++) {
            MultiLabelOutput ensembleMLO = ensemble[i].makePrediction(instance);
            boolean[] bip = ensembleMLO.getBipartition();
            double[] conf = ensembleMLO.getConfidences();

            for (int j = 0; j < numLabels; j++) {
                sumVotes[j] += bip[j] == true ? 1 : 0;
                sumConf[j] += conf[j];
            }
        }

        double[] confidence = new double[numLabels];
        for (int j = 0; j < numLabels; j++) {
            if (useConfidences) {
                confidence[j] = sumConf[j] / numOfModels;
            } else {
                confidence[j] = sumVotes[j] / (double) numOfModels;
            }
        }
        
        MultiLabelOutput mlo;
        if(!useFmeasureOptimizationThreshold){
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
	
    //calculate the thresholds to distinguish the relative and irrelative instances of each label based on optimizaiton of Macro F-measure
    protected void calculateThresholds(MultiLabelInstances trainingSet) throws Exception{
    	
    	thresholds=new double[numLabels];
    	
    	this.useFmeasureOptimizationThreshold=false;
    	double predictConfidences[][]=new double [trainingSet.getNumInstances()][trainingSet.getNumLabels()];
    	for(int i=0;i<trainingSet.getNumInstances();i++){
    		Instance data=trainingSet.getDataSet().get(i);
    		MultiLabelOutput mlo=this.makePredictionInternal(data);
    		predictConfidences[i]=Arrays.copyOf(mlo.getConfidences(),numLabels); 
    	}
    	this.useFmeasureOptimizationThreshold=true;
    	     
    	for(int j=0;j<numLabels;j++){
    		double maxF=Double.MIN_VALUE;
    		boolean truelabels[]=new boolean[trainingSet.getNumInstances()];
    		for(int i=0;i<trainingSet.getNumInstances();i++){
        		truelabels[i]=trainingSet.getDataSet().get(i).stringValue(labelIndices[j]).equals("1");
        	}
        	for(double d=0.05D;d<1.0D;d+=0.05D){
        		int tp=0,fp=0,fn=0;
        		for(int i=0;i<trainingSet.getNumInstances();i++){
        			boolean preidctLabel=predictConfidences[i][j]>=d;
        			if(preidctLabel){
        				if(truelabels[i])
        					tp++;
        				else
        					fp++;
        			}
        			else{
        				if(truelabels[i]){
        					fn++;
        				}
        			}
        		}				 
        		double f=InformationRetrievalMeasures.fMeasure(tp,fp,fn,1.0);
        		if(f>maxF){
        			maxF=f;
        			thresholds[j]=d;
        		}
        	}
    	}  
    }
    

}
