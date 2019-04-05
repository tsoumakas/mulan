package mulan.classifier.ensemble;

import java.util.Arrays;

import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.measure.InformationRetrievalMeasures;
import mulan.sampling.MultiLabelSampling;
import mulan.sampling.MLSOL;
import weka.core.Instance;
import weka.core.TechnicalInformation;


/**
 * Implementation of EMLS.</p> 
 * 
 *
 * @author Bin Liu
 * @version 2019.3.21
 * 
 */

public class EnsembleOfSampling extends HomogeneousEnsembleMultiLabelLearner{
	protected MultiLabelLearner mlls[];
	protected MultiLabelSampling mlsampling=new MLSOL();
	
	protected int numModels=5; // number of learner (ensemble size)
	protected double P=0.3;  //sampling ratio
	protected int seed=1;
	
	
	protected double thresholds[]=null;
	public enum thresholdOptimizationMeasures{
		None,
		Fmeasure,
		Gmean,
		BalancedAcuraccy,
	}
	public thresholdOptimizationMeasures measure=thresholdOptimizationMeasures.Fmeasure;
	
	/**
	 * @return the mlsampling
	 */
	public MultiLabelSampling getMlsampling() {
		return mlsampling;
	}

	/**
	 * @param mlsampling the mlsampling to set
	 */
	public void setMlsampling(MultiLabelSampling mlsampling) {
		this.mlsampling = mlsampling;
	}
	
	
	/**
	 * @return the p
	 */
	public double getP() {
		return P;
	}

	/**
	 * @return the numModels
	 */
	public int getNumModels() {
		return numModels;
	}

	/**
	 * @param p the p to set
	 */
	public void setP(double p) {
		P = p;
	}

	/**
	 * @param numModels the numModels to set
	 */
	public void setNumModels(int numModels) {
		this.numModels = numModels;
	}
	
	@Override
	protected void buildInternal(MultiLabelInstances trainingSet) throws Exception {
		mlls=new MultiLabelLearner[numModels];
		mlsampling.setP(P);
		for(int i=0;i<numModels;i++){
			debug("Model"+(i+1)+" Sampling");
			mlsampling.setSeed(i+seed);
			MultiLabelInstances mlData=mlsampling.build(trainingSet);			
			//debug("Model"+(i+1)+" Training");
			mlls[i]=baseMlLearner.makeCopy();
			mlls[i].build(mlData);
		}
		
		if(measure!=thresholdOptimizationMeasures.None){
			debug("Calculating thresholds");
			calculateThresholds(trainingSet);         	
        }
	}

	@Override
	protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception, InvalidDataException {
		double conf[]=new double[numLabels];
		for(int i=0;i<numModels;i++){
			MultiLabelOutput mlo=mlls[i].makePrediction(instance);
			double ds[]=mlo.getConfidences();
			for(int j=0;j<numLabels;j++){
				conf[j]+=ds[j];
			}
		}
		for(int j=0;j<numLabels;j++){
			conf[j]/=numModels;
		}
		
		MultiLabelOutput mlo;
		if(measure==thresholdOptimizationMeasures.None){
        	mlo = new MultiLabelOutput(conf, 0.5);
        }
        else{
        	boolean bipartition[]=new boolean[numLabels];
        	for(int j=0;j<numLabels;j++){
        		bipartition[j]=conf[j]>=thresholds[j];
        	}
        	mlo=new MultiLabelOutput(bipartition, conf);
        }
		return mlo;
	}

	@Override
	public TechnicalInformation getTechnicalInformation() {
		// TODO Auto-generated method stub
		return null;
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
        
	
}
