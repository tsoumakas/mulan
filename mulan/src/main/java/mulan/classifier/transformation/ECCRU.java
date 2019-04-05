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
import mulan.classifier.MultiLabelOutput;
import mulan.data.ImbalancedStatistics;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.measure.InformationRetrievalMeasures;
import mulan.classifier.transformation.ClassifierChainUnderSampling;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

/**
 * <p>Implementation of the ECCRU algorithm.</p> 
 * <p>For more information, see <em> Liu, Bin Tsoumakas, Grigorios. 
 * "Making Classifier Chains Resilient to Class Imbalance." ACML. 2018. pp.280-295</em></p>
 *
 * @author Bin Liu
 * @version 2018.5.10
 */

public class ECCRU extends EnsembleOfClassifierChains {
   
	public enum thresholdOptimizationMeasures{
		None,
		Fmeasure,
		Gmean,
		BalancedAcuraccy,
	}
	protected thresholdOptimizationMeasures measure=thresholdOptimizationMeasures.Fmeasure;


	
    /**
     * The under Sampling Percent of ClassiferChainUnderSampling, it will be used when useClassiferChainUnderSampling is true
     **/
    protected double underSamplingPercent=1.0;
    
    protected double thresholds[]=null;

	public double getUnderSamplingPercent() {
		return underSamplingPercent;
	}

	public void setUnderSamplingPercent(double underSamplingPercent) {
		this.underSamplingPercent = underSamplingPercent;
	}



	/**
	 * @return the measure
	 */
	public thresholdOptimizationMeasures getMeasure() {
		return measure;
	}


	/**
	 * @param measure the measure to set
	 */
	public void setMeasure(thresholdOptimizationMeasures measure) {
		this.measure = measure;
	}
    
    /**
     * Default constructor
     */
    public ECCRU() {
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
    public ECCRU(Classifier classifier, int aNumOfModels,
            boolean doUseConfidences, boolean doUseSamplingWithReplacement) {
        super(classifier,aNumOfModels,doUseConfidences,doUseSamplingWithReplacement);
    }

    
  
    
    @Override
    protected void buildInternal(MultiLabelInstances trainingSet) throws Exception {
        Instances dataSet = new Instances(trainingSet.getDataSet());
        Random rand=new Random(seed);
        
        for (int i = 0; i < numOfModels; i++) {
            debug("ECC Building Model:" + (i + 1) + "/" + numOfModels);
            MultiLabelInstances train=null;
            Instances sampledDataSet=null;
            dataSet.randomize(new Random(rand.nextInt()));
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
            train = new MultiLabelInstances(sampledDataSet, trainingSet.getLabelsMetaData());

            
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
            ensemble[i] = new ClassifierChainUnderSampling(baseClassifier, chain,underSamplingPercent); 
            ensemble[i].build(train);
        }
        
       
        if(measure!=thresholdOptimizationMeasures.None){
        	calculateThresholds(trainingSet);         	
        }
    }
    
    //calculate the thresholds to distinguish the relevant and irrelevant instances of each label based on optimization of Macro F-measure
    protected void calculateThresholds(MultiLabelInstances trainingSet) throws Exception{
    	thresholds=new double[numLabels];
    	thresholdOptimizationMeasures m=measure;
    	
    	measure=thresholdOptimizationMeasures.None;
    	double predictConfidences[][]=new double [trainingSet.getNumInstances()][trainingSet.getNumLabels()];
    	for(int i=0;i<trainingSet.getNumInstances();i++){
    		Instance data=trainingSet.getDataSet().get(i);
    		MultiLabelOutput mlo=this.makePredictionInternal(data);
    		predictConfidences[i]=Arrays.copyOf(mlo.getConfidences(),numLabels); 
    	}
    	measure=m;
    	     
    	for(int j=0;j<numLabels;j++){
    		double max=Double.MIN_VALUE;
    		boolean truelabels[]=new boolean[trainingSet.getNumInstances()];
    		for(int i=0;i<trainingSet.getNumInstances();i++){
        		truelabels[i]=trainingSet.getDataSet().get(i).stringValue(labelIndices[j]).equals("1");
        	}
        	for(double d=0.05D;d<1.0D;d+=0.05D){
        		int tp=0,tn=0,fp=0,fn=0;
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
        				else{
        					tn++;
        				}
        			}
        		}				 
        		double value=0;
        		switch (measure){
        			case Fmeasure:
        				value=InformationRetrievalMeasures.fMeasure(tp,fp,fn,1.0); break;
        			case Gmean:
        				value=InformationRetrievalMeasures.gMean(tp, tn, fp, fn); break;
        			case BalancedAcuraccy:
        				value=InformationRetrievalMeasures.balancedAccuracy(tp, tn, fp, fn); break;
        		}
        			
        		if(value>max){
        			max=value;
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