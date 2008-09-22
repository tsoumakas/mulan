package mulan.classifier;
import weka.core.Utils;



/**
 * Simple container class for multilabel classification result 
 */
public class Prediction {

	protected double[] confidences;
	protected double[] predictedLabels;
		
	public double[] getConfidences()
	{
		return confidences;
	}
	
	public double[] getPredictedLabels()
	{
		return predictedLabels;
	}
	
	public boolean getPrediction(int label)
	{
		return Utils.eq(1, predictedLabels[label]);
	}
	
	/**
	 * Is confidence a proper name?
	 * @param label
	 * @return
	 */
	public double getConfidence(int label)
	{
		return confidences[label];
	}
	
	public Prediction(double[] labels, double[] confidences)
	{
		this.confidences = confidences;
		predictedLabels = labels;
	}
	
	/**
	 * Number of predicted labels for this instance. 
	 * Calculated only once.
	 */
	protected int numLabels = -1;
	
	/**
	 * Number of predicted labels for this instance.
	 */
	public int numLabels()
	{
		if (numLabels == -1) numLabels =(int) Utils.sum(predictedLabels);
		return numLabels; 
	}

	/**
	 * String representation of the set of labels. Perhaps we
	 * could obtain the actual attribute names from somewhere?
	 */
    @Override
	public String toString()
	{
		StringBuilder str = new StringBuilder().append("{");
		

		str.append("}");
		return str.toString();
	}
	
}
