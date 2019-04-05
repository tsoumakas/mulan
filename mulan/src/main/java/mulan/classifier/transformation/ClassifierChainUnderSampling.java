package mulan.classifier.transformation;

import mulan.data.MultiLabelInstances;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.supervised.instance.SpreadSubsampleWithMissClassValues;
import weka.filters.unsupervised.attribute.Remove;

/**
 * <p>Implementation of the CCRU algorithm.</p> 
 * <p>For more information, see <em> Liu, Bin Tsoumakas, Grigorios. 
 * "Making Classifier Chains Resilient to Class Imbalance." ACML. 2018.</em></p>
 *
 * @author Bin Liu
 * @version 2018.12.19
 */

public class ClassifierChainUnderSampling extends ClassifierChain{

	protected int seed=1;
	protected double underSamplingPercent=1.0;  //percentage of majority instances to be deleted 
		
	public double getUnderSamplingPercent() {
		return underSamplingPercent;
	}

	public void setUnderSamplingPercent(double p) {
		underSamplingPercent = p;
	}
	
	/**
	 * @return the seed
	 */
	public int getSeed() {
		return seed;
	}

	/**
	 * @param seed the seed to set
	 */
	public void setSeed(int seed) {
		this.seed = seed;
	}

	
	public ClassifierChainUnderSampling(){
		 super(new J48());
	}
	
	public ClassifierChainUnderSampling(Classifier classifier){
		 super(classifier);
	}
	
	public ClassifierChainUnderSampling(Classifier classifier, int[] aChain) {
        super(classifier);
        chain = aChain;
    }
	
	public ClassifierChainUnderSampling(Classifier classifier, int[] aChain,double underSamplingPercent) {
        this(classifier,aChain);
        this.underSamplingPercent=underSamplingPercent;
    }
	
	
    protected void buildInternal(MultiLabelInstances train) throws Exception {
    	
    	if (chain == null) {
            chain = new int[numLabels];
            for (int i = 0; i < numLabels; i++) {
                chain[i] = i;
            }
        }

        numLabels = train.getNumLabels();
        ensemble = new FilteredClassifier[numLabels];
        
        Instances trainCopy = new Instances(train.getDataSet());  //copy of the training data set
        
        for (int modelIndex = 0; modelIndex < numLabels; modelIndex++) {
            ensemble[modelIndex] = new FilteredClassifier();
            ensemble[modelIndex].setClassifier(AbstractClassifier.makeCopy(baseClassifier));

            // Indices of attributes to remove first removes numLabels attributes
            // the numLabels - 1 attributes and so on.
            // The loop starts from the last attribute.
            int[] indicesToRemove = new int[numLabels - 1 - modelIndex];
            int counter2 = 0;
            for (int counter1 = 0; counter1 < numLabels - modelIndex - 1; counter1++) {
                indicesToRemove[counter1] = labelIndices[chain[numLabels - 1 - counter2]];
                counter2++;
            }

            Remove remove = new Remove();
            remove.setAttributeIndicesArray(indicesToRemove);
            remove.setInputFormat(trainCopy);
            remove.setInvertSelection(false);
            ensemble[modelIndex].setFilter(remove);

            trainCopy.setClassIndex(labelIndices[chain[modelIndex]]);
            debug("Bulding model " + (modelIndex + 1) + "/" + numLabels);
            
            //Undersample majority instances randomly
            int c0=0,c1=0;
            for(Instance data:trainCopy){
            	if(data.stringValue(data.classIndex()).equals("1")){
            		c1++;
            	}
            	else if(data.stringValue(data.classIndex()).equals("0")){
            		c0++;
            	}
            }
            
            Instances sampledTrainSet=trainCopy;
            int minNum=Math.min(c0, c1), maxNum=Math.max(c0, c1);
            double d=maxNum*(1.0-underSamplingPercent)/minNum;
            if(d<1.0){
            	d=1.0;
            }
            SpreadSubsampleWithMissClassValues ss=new SpreadSubsampleWithMissClassValues();
            ss.setRandomSeed(seed);
            ss.setDistributionSpread(d);
            ss.setInputFormat(trainCopy);  
            sampledTrainSet = ss.useFilter(trainCopy, ss);  
            
           
            ensemble[modelIndex].buildClassifier(sampledTrainSet);
            
            
            for(int i=0;i<trainCopy.numInstances();i++){
            	Instance data=trainCopy.get(i);
            	double distribution[];
                distribution = ensemble[modelIndex].distributionForInstance(data);
                int maxIndex = (distribution[0] > distribution[1]) ? 0 : 1;
                data.setValue(labelIndices[chain[modelIndex]], maxIndex);
            }

        }
    }
    
 
}
