package mulan.classifier;
import java.util.Vector;

import mulan.core.LabelSet;
import weka.core.Instance;
import weka.core.Instances;


@SuppressWarnings("serial")
public class HybridSubsetMapper extends SubsetMapper
{
	private LabelPowersetClassifier classifier;
	
	
	public HybridSubsetMapper(Instances instances, int numLabels, int diff)
	throws Exception
	{
		this(instances, numLabels);
		distanceThreshold = diff;
	}
	
    public HybridSubsetMapper(Instances instances, int numLabels)
    throws Exception
    {
    	super(instances, numLabels);

    	/*
    	LibSVM lsBaseClassifier = new LibSVM();
    	svm.setOptions(Utils.splitOptions(
    		"-Z"
    	));
    	svm.setProbabilityEstimates(true);
    	//*/
    	//*
    	weka.classifiers.lazy.IBk lsBaseClassifier = new weka.classifiers.lazy.IBk();
    	lsBaseClassifier.setKNN(10);
    	//*/
    	classifier = new LabelPowersetClassifier(lsBaseClassifier, numLabels);
    	classifier.buildClassifier(instances);
    	
    }
    
    public Prediction nearestSubset(Instance instance, double[] labels)
    throws Exception
    {
    	LabelSet set = new LabelSet(labels);
    	Vector<LabelSet> neighbors = subsetsWithinDiff(set, distanceThreshold);
    	if (neighbors.size() == 0) 
    		return new Prediction(labels, this.calculateConfidences(set));
    	
    	//Had to hack into LabelPowersetClassifier to expose this information
    	double[] distro = classifier.distributionFromBaseClassifier(instance);
    	
    	//Find the subset with the highest prior probability
    	LabelSet nearest = null;
    	double bestProb = Double.MIN_VALUE;
    	for(LabelSet neighbor : neighbors)
    	{
    		int labelIdx = classifier.indexOfClassValue(neighbor.toBitString());
    		if (distro[labelIdx] > bestProb)
    		{
    			bestProb = distro[labelIdx];
    			nearest  = neighbor;
    		}
    	}
        if (nearest == null)
            return new Prediction(labels, this.calculateConfidences(set));
        
    	return new Prediction(nearest.toDoubleArray(), calculateConfidences(nearest));

    }
}
