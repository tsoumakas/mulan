/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */
package mulan.classifier.transformation;

import java.util.Arrays;
import java.util.Random;
import mulan.classifier.InvalidDataException;
import mulan.classifier.ModelInitializationException;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.measure.InformationRetrievalMeasures;
import mulan.sampling.ClassifierChainUnderSampling;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

/**
 * <p>Implementation of the Ensemble of Classifier Chains(ECC) algorithm.</p>
 * <p>For more information, see <em>Read, J.; Pfahringer, B.; Holmes, G., Frank,
 * E. (2011) Classifier Chains for Multi-label Classification. Machine Learning.
 * 85(3):335-359.</em></p>
 *
 * @author Eleftherios Spyromitros-Xioufis
 * @author Konstantinos Sechidis
 * @author Grigorios Tsoumakas
 * @version 2012.02.27
 */
public class EnsembleOfClassifierChains extends TransformationBasedMultiLabelLearner {

    /**
     * The number of classifier chain models
     */
    protected int numOfModels;
    /**
	 * @return the numOfModels
	 */
	public int getNumOfModels() {
		return numOfModels;
	}



	/**
     * An array of ClassifierChain models
     */
    protected ClassifierChain[] ensemble;
    /**
     * Random number generator
     */
    protected Random rand;
    /**
     * Whether the output is computed based on the average votes or on the
     * average confidences
     */
    protected boolean useConfidences;
    /**
     * Whether to use sampling with replacement to create the data of the models
     * of the ensemble
     */
    protected boolean useSamplingWithReplacement = true;
    /**
     * The size of each bag sample, as a percentage of the training size. Used
     * when useSamplingWithReplacement is true
     */
    protected int BagSizePercent = 100;

    /**
     * The size of each sample, as a percentage of the training size Used when
     * useSamplingWithReplacement is false
     */
    protected double samplingPercentage = 67;
    
    
    /**
     * Whether use the ClassiferChainUnderSampling instead of ClassiferChain in the model
     */
    protected boolean useClassiferChainUnderSampling=false;
    
    /**
     * The under Sampling Percent of ClassiferChainUnderSampling, it will be used when useClassiferChainUnderSampling is true
     **/
    protected double underSamplingPercent=1.0;
    
    
	protected boolean useFmeasureOptimizationThreshold=false;
    
    double thresholds[]=null;
    

	public double getUnderSamplingPercent() {
		return underSamplingPercent;
	}


	public void setUnderSamplingPercent(double underSamplingPercent) {
		this.underSamplingPercent = underSamplingPercent;
	}


    /**
	 * @return the useFmeasureOptimizationThreshold
	 */
	public boolean isUseFmeasureOptimizationThreshold() {
		return useFmeasureOptimizationThreshold;
	}

	/**
	 * @param useFmeasureOptimizationThreshold the useFmeasureOptimizationThreshold to set
	 */
	public void setUseFmeasureOptimizationThreshold(boolean useFmeasureOptimizationThreshold) {
		this.useFmeasureOptimizationThreshold = useFmeasureOptimizationThreshold;
	}

	/**
	 * @return the useClassiferChainUnderSampling
	 */
	public boolean isUseClassiferChainUnderSampling() {
		return useClassiferChainUnderSampling;
	}

	/**
	 * @param useClassiferChainUnderSampling the useClassiferChainUnderSampling to set
	 */
	public void setUseClassiferChainUnderSampling(boolean useClassiferChainUnderSampling) {
		this.useClassiferChainUnderSampling = useClassiferChainUnderSampling;
	}

	
	
	
	/**
     * Returns the size of each bag sample, as a percentage of the training size
     *
     * @return the size of each bag sample, as a percentage of the training size
     */
    public int getBagSizePercent() {
        return BagSizePercent;
    }

    /**
     * Sets the size of each bag sample, as a percentage of the training size
     *
     * @param bagSizePercent the size of each bag sample, as a percentage of the
     * training size
     */
    public void setBagSizePercent(int bagSizePercent) {
        BagSizePercent = bagSizePercent;
    }

    /**
     * Returns the sampling percentage
     *
     * @return the sampling percentage
     */
    public double getSamplingPercentage() {
        return samplingPercentage;
    }

    /**
     * Sets the sampling percentage
     *
     * @param samplingPercentage the sampling percentage
     */
    public void setSamplingPercentage(double samplingPercentage) {
        this.samplingPercentage = samplingPercentage;
    }


    /**
     * Default constructor
     */
    public EnsembleOfClassifierChains() {
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
    public EnsembleOfClassifierChains(Classifier classifier, int aNumOfModels,
            boolean doUseConfidences, boolean doUseSamplingWithReplacement) {
        super(classifier);
        numOfModels = aNumOfModels;
        useConfidences = doUseConfidences;
        useSamplingWithReplacement = doUseSamplingWithReplacement;
        ensemble = new ClassifierChain[aNumOfModels];
        rand = new Random(1);
    }

    @Override
    protected void buildInternal(MultiLabelInstances trainingSet) throws Exception {

        Instances dataSet = new Instances(trainingSet.getDataSet());

        for (int i = 0; i < numOfModels; i++) {
            debug("ECC Building Model:" + (i + 1) + "/" + numOfModels);
            Instances sampledDataSet;
            dataSet.randomize(rand);
            
            if (useSamplingWithReplacement) {
                int bagSize = dataSet.numInstances() * BagSizePercent / 100;
                // create the in-bag dataset
                sampledDataSet = dataSet.resampleWithWeights(new Random(1));
                if (bagSize < dataSet.numInstances()) {
                    sampledDataSet = new Instances(sampledDataSet, 0, bagSize);
                }
            } else {
                RemovePercentage rmvp = new RemovePercentage();
                rmvp.setInvertSelection(true);
                rmvp.setPercentage(samplingPercentage);
                rmvp.setInputFormat(dataSet);
                sampledDataSet = Filter.useFilter(dataSet, rmvp);
            }
            
            
            MultiLabelInstances train = new MultiLabelInstances(sampledDataSet, trainingSet.getLabelsMetaData());

            int[] chain = new int[numLabels];
            for (int j = 0; j < numLabels; j++) {
                chain[j] = j;
            }
            for (int j = 0; j < chain.length; j++) {
                int randomPosition = rand.nextInt(chain.length);
                int temp = chain[j];
                chain[j] = chain[randomPosition];
                chain[randomPosition] = temp;
            }
            debug(Arrays.toString(chain));

            // MAYBE WE SHOULD CHECK NOT TO PRODUCE THE SAME VECTOR FOR THE
            // INDICES
            // BUT IN THE PAPER IT DID NOT MENTION SOMETHING LIKE THAT
            // IT JUST SIMPLY SAY A RANDOM CHAIN ORDERING OF L
            if(useClassiferChainUnderSampling)
            	ensemble[i] = new ClassifierChainUnderSampling(baseClassifier, chain,underSamplingPercent);
            else
            	ensemble[i]= new ClassifierChain(baseClassifier, chain);
            ensemble[i].build(train);
        }
        
       
        if(useFmeasureOptimizationThreshold){
        	calculateThresholds(trainingSet);         	
        }
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
    
    
    
    @Override
    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception,
            InvalidDataException {

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
}