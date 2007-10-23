package mulan.evaluation;
import weka.core.Utils;





/**
 * Calculate measures separately for each label and  average the results. Macroaveraging.
 * @author  rofr
 */
public class LabelBasedEvaluation extends EvaluationBase
{

	/**
	 * Constant used as array index to retrieve.
	 */
	public final static int MACRO = 0;
	public final static int MICRO = 1;
	

	//TODO: is this the appropriate default?
	protected final static int DEFAULT = MACRO; 
	
	/**
	 * Keep track of which type of measure to return.
	 */
	protected int averagingMethod;
	
	/**
	 * store both micro and macro averaged measures.
	 */
	protected double[] recall    = new double[2];
	protected double[] precision = new double[2];
	protected double[] fmeasure  = new double[2];
	
	//I think we should leave this measure for completeness even
	//if it is equivalent to hammingloss. Perhaps a user will choose
	//either label or example based for a specific scenario.
	protected double[] accuracy  = new double[2];
	
	
	//Per label measures
	protected double[] labelAccuracy;
	protected double[] labelRecall;  
	protected double[] labelPrecision;
	protected double[] labelFmeasure;
	
	public double accuracy(int label)
	{
		return labelAccuracy[label];
	}

	public double recall(int label)
	{
		return labelRecall[label];
	}
	
	public double precision(int label)
	{
		return labelPrecision[label];
	}
	
	public double fmeasure(int label)
	{
		return labelFmeasure[label];
	}
	
	protected LabelBasedEvaluation(BinaryPrediction[][] predictions)
	throws Exception
	{
		this(predictions, DEFAULT);
	}
	
	/**
	 * Used by crossvalidation subclass.
	 *
	 */
	protected LabelBasedEvaluation()
	{
		super(null);
	}
	
	protected LabelBasedEvaluation(BinaryPrediction[][] predictions, int averagingMethod)
	throws Exception
	{
		super(predictions);
		setAveragingMethod(averagingMethod);
		computeMeasures();
	}

	/**
	 * Compute both micro and macro averages.
	 */
	protected void computeMeasures()
	{
            int numInstances = numInstances();
            int numLabels   = numLabels();

            //Counters are doubles to avoid typecasting
            //when performing divisions. It makes the code a
            //little cleaner but:
            //TODO: run performance tests on counting with doubles
            double[] falsePositives    = new double[numLabels];
            double[] truePositives     = new double[numLabels];
            double[] falseNegatives    = new double[numLabels];
            double[] trueNegatives     = new double[numLabels];

            this.labelAccuracy         = new double[numLabels];
            this.labelRecall           = new double[numLabels];
            this.labelPrecision        = new double[numLabels];
            this.labelFmeasure          = new double[numLabels];


            //Count TP, TN, FP, FN
            for(int i = 0; i < numInstances; i++)
            {
                    for(int j = 0; j < numLabels; j++)
                    {	
                            boolean actual = predictions[i][j].actual;
                            boolean predicted = predictions[i][j].predicted;

                            if (actual && predicted) truePositives[j]++;
                            else if (!actual && !predicted) trueNegatives[j]++;
                            else if (predicted) falsePositives[j]++;
                            else falseNegatives[j]++;
                    }
            }

            //Compute macro averaged measures
            for(int i = 0; i < numLabels; i++)
            {
                    labelAccuracy[i] = (truePositives[i] + trueNegatives[i]) / numInstances; 

                    labelRecall[i] = truePositives[i] + falseNegatives[i] == 0 ? 0
                                    :truePositives[i] / (truePositives[i] + falseNegatives[i]);

                    labelPrecision[i] = truePositives[i] + falsePositives[i] == 0 ? 0
                                    :truePositives[i] / (truePositives[i] + falsePositives[i]);

                    labelFmeasure[i] = computeFMeasure(labelPrecision[i], labelRecall[i]);
            }
            this.accuracy[MACRO]  = Utils.mean(labelAccuracy);
	    this.recall[MACRO]    = Utils.mean(labelRecall);
	    this.precision[MACRO] = Utils.mean(labelPrecision);
	    this.fmeasure[MACRO]  = Utils.mean(labelFmeasure);
	    
	    //Compute micro averaged measures
	    double tp = Utils.sum(truePositives);
	    double tn = Utils.sum(trueNegatives);
	    double fp = Utils.sum(falsePositives);
	    double fn = Utils.sum(falseNegatives);
	    
	    this.accuracy[MICRO]  = (tp + tn) / (numInstances * numLabels);
	    this.recall[MICRO]    = tp + fn == 0 ? 0 : tp / (tp + fn);
	    this.precision[MICRO] = tp + fp == 0 ? 0 : tp / (tp + fp);
	    this.fmeasure[MICRO]  = computeFMeasure(this.precision[MICRO], this.recall[MICRO]);
	}

	/**
	 * @param averagingMethod must be one of MICROAVERAGED or MACROAVERAGED
	 *
	 */
	//TODO: Use Enum instead...
	public void setAveragingMethod(int averagingMethod)
	{
		this.averagingMethod = averagingMethod;
	}

	/**
	 * @return the averagingMethod
	 */
	public int getAveragingMethod()
	{
		return averagingMethod;
	}

	@Override
	public double accuracy()
	{
		return accuracy[averagingMethod];
	}

	@Override
	public double fmeasure()
	{
		return fmeasure[averagingMethod];
	}

	@Override
	public double precision()
	{
		return precision[averagingMethod];
	}

	@Override
	public double recall()
	{
		return recall[averagingMethod];
	}
}
