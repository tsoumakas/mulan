package mulan.classifier;
import mulan.core.Statistics;
import mulan.core.*;
import weka.core.*;

import java.io.Serializable;
import java.util.*;

/*
 * Maps a predicted set of labels to the nearest set present 
 * in the training data based on hamming difference.
 */
class SubsetMapper implements Serializable
{
   
	private static final long serialVersionUID = -8083409997373802735L;

	/**
     * All individual label subsets and a count of the number of 
     * instances of each.
     */
    private HashMap<LabelSet, Integer> labelSubsetCount;

    /**
     * We will probably need a reference to this later. I feel
     * this type should be named Characteristics.
     */
    protected Statistics statistics;
       
    
    /**
     * Dont change the prediction unless the difference is
     * less than or equal to this value.
     */
    protected int distanceThreshold = Integer.MAX_VALUE;
    
    public SubsetMapper(Instances instances, int numLabels)
    {
    	statistics  = new Statistics();
        statistics.calculateStats(instances, numLabels);
    	labelSubsetCount = statistics.labelCombCount();
    }
    
    public SubsetMapper(Instances instances, int numLabels, int distanceThreshold)
    {
    	this(instances, numLabels);
    	this.distanceThreshold = distanceThreshold;
    }
    
    /**
     * 
     * @param labels
     * @return
     */
    public Prediction nearestSubset(Instance instance, double[] labels)
    throws Exception
    {
    	LabelSet set = new LabelSet(labels);
    	LabelSet nearest = null;

    	//Almost missed this case!
    	if (labelSubsetCount.containsKey(set))
    	{
    		return new Prediction(set.toDoubleArray(), 
    							  calculateConfidences(set));
    	}

    	int closestCount = 0;
        int minDistance = Integer.MAX_VALUE;
        for(LabelSet current : shuffle(labelSubsetCount.keySet()))
        {
            int distance = set.hammingDifference(current);
            if (distance == minDistance)
            {
                int count = labelSubsetCount.get(current);
                if (count > closestCount)
                {
                    nearest = current;
                    closestCount = count;                    
                }
            }
            if (distance < minDistance)
            {
                minDistance = distance;
                nearest = current;
                closestCount = labelSubsetCount.get(nearest);
            }
        } 
        if (minDistance <= distanceThreshold)
        	return new Prediction(nearest.toDoubleArray(), calculateConfidences(nearest));
        else
        	return new Prediction(labels, calculateConfidences(set));
    }
    
    protected Vector<LabelSet> subsetsWithinDiff(LabelSet set, int diff)
    {
    	Vector<LabelSet> result = new Vector<LabelSet>();
    	for(LabelSet candidate : labelSubsetCount.keySet())
    	{
    		if(candidate.hammingDifference(set) <= diff)
    			result.add(candidate);
    	}
    	return result;
    }
    
    private Collection<LabelSet> shuffle(Set<LabelSet> labelSubsets)
    {
    	int seed = 1;
    	Vector<LabelSet> result = new Vector<LabelSet>(labelSubsets.size());
    	result.addAll(labelSubsets);
    	Collections.shuffle(result, new Random(seed));
    	return result;
    }
    
    protected double[] calculateConfidences(LabelSet set)
    {
    	return set.toDoubleArray(); 
    }
}

