package mulan.sampling;

import java.util.Enumeration;

import mulan.classifier.transformation.ClassifierChain;
import mulan.data.MultiLabelInstances;
import mulan.transformations.regression.ChainTransformation;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.supervised.instance.SpreadSubsample;
import weka.filters.unsupervised.attribute.Remove;

public class ClassifierChainUnderSampling extends ClassifierChain{
	
	protected double underSamplingPercent=0.3;  //percentage of majority instances to be deleted 
	
    /**
     * The training data of each classifier of the chain. After training the actual data are deleted and only
     * the header information is held which is needed during prediction.
     */
    private Instances[] chainClassifierTrainSets;
	
	
	public double getUnderSamplingPercent() {
		return underSamplingPercent;
	}


	public void setUnderSamplingPercent(double p) {
		underSamplingPercent = p;
	}
	
	
	public ClassifierChainUnderSampling(){
		 super(new J48());
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

        Instances trainDataset;
        numLabels = train.getNumLabels();
        ensemble = new FilteredClassifier[numLabels];
        //trainDataset = train.getDataSet();
        
        Instances trainCopy = new Instances(train.getDataSet());  //copy of the training data set
        
        for (int modelIndex = 0; modelIndex < numLabels; modelIndex++) {
            ensemble[modelIndex] = new FilteredClassifier();
            ensemble[modelIndex].setClassifier(AbstractClassifier.makeCopy(baseClassifier));

            //chainClassifierTrainSets[i] = ChainTransformation.transformInstances(trainCopy, chain, i + 1);
            
            
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
            
            //Under sample majority instances randomly
            int c0=0,c1=0;
            for(Instance data:trainCopy){
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
            
            
            //System.out.print("label "+labelIndices[modelIndex]+": "+c0+"\t"+c1+"\t");
            SpreadSubsample ss=new SpreadSubsample();
            //ss.setMaxCount(minNum); //the number of minority class instances 
            ss.setDistributionSpread(d);
            ss.setInputFormat(trainCopy);  
            Instances strainDataset = ss.useFilter(trainCopy, ss);  
            
            
            /*
            c0=c1=0;
            for(Instance data:strainDataset){
            	if(data.stringValue(data.classIndex()).equals("1")){
            		c1++;
            	}
            	else{
            		c0++;
            	}
            }
            System.out.print("label "+labelIndices[modelIndex]+": "+c0+"\t"+c1+"\t");
            */
            
            ensemble[modelIndex].buildClassifier(strainDataset);
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
