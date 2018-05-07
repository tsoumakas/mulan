package mulan.classifier.transformation;

import mulan.data.MultiLabelInstances;
import mulan.transformations.BinaryRelevanceTransformation;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.supervised.instance.SpreadSubsampleWithMissClassValues;


public class BinaryRelevanceUnderSampling extends BinaryRelevance{
	
	/**
	 * Percentage of majority class instances for each label to be deleted
	 * The 1.0 value denotes that the equal size of the majority and minority instances after under sampling 
	 */
	protected double underSamplingPercent=1.0;  
	
	protected int seed=1;
	

	public BinaryRelevanceUnderSampling(Classifier classifier) {
		super(classifier);
	}
	
	public BinaryRelevanceUnderSampling(Classifier classifier,double underSamplingPercent) {
		super(classifier);
		this.underSamplingPercent=underSamplingPercent;
	}
	
	public BinaryRelevanceUnderSampling(Classifier classifier,double underSamplingPercent,int seed) {
		super(classifier);
		this.underSamplingPercent=underSamplingPercent;
		this.seed=seed;
	}
	
	
	public double getUnderSamplingPercent() {
		return underSamplingPercent;
	}
	
	public void setUnderSamplingPercent(double underSamplingPercent) {
		this.underSamplingPercent = underSamplingPercent;
	}

	public long getSeed() {
		return seed;
	}


	public void setSeed(int seed) {
		this.seed = seed;
	}    
	

    protected void buildInternal(MultiLabelInstances train) throws Exception {
        ensemble = new Classifier[numLabels];

        correspondence = new String[numLabels];
        for (int i = 0; i < numLabels; i++) {
            correspondence[i] = train.getDataSet().attribute(labelIndices[i]).name();
        }

        debug("preparing shell");
        brt = new BinaryRelevanceTransformation(train);

        for (int i = 0; i < numLabels; i++) {
            ensemble[i] = AbstractClassifier.makeCopy(baseClassifier);
            Instances shell = brt.transformInstances(i);
            debug("Bulding model " + (i + 1) + "/" + numLabels);            
            
            int c0=0,c1=0;
            for(Instance data:shell){
            	if(data.stringValue(data.classIndex()).equals("1")){
            		c1++;
            	}
            	else{
            		c0++;
            	}
            }
            int minNum=Math.min(c0, c1), maxNum=Math.max(c0, c1);
            double d=maxNum*(1.0-underSamplingPercent)/minNum;
            if(d<1.0){
            	d=1.0;
            }
            
            
            
            SpreadSubsampleWithMissClassValues ss=new SpreadSubsampleWithMissClassValues();
            ss.setRandomSeed(seed);
            ss.setDistributionSpread(d);
            ss.setInputFormat(shell);  
            Instances sshell = ss.useFilter(shell, ss);  
            
            
            ensemble[i].buildClassifier(sshell);
        }
    }

}
