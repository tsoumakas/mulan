package mulan.evaluation;
/**
 * Data holding structure to make evaluation computation a little cleaner. 
 * Note that the confidence, refers to the confidence of the result being true,
 * and not the confidence of the result itself which can be true or false. 
 */
public class BinaryPrediction
{
    public BinaryPrediction(boolean predicted, boolean actual, double confidenceTrue)
    {
            this.predicted = predicted;
            this.actual = actual;
            this.confidenceTrue = confidenceTrue;
    }

    protected boolean actual;
    protected boolean predicted;
    protected double confidenceTrue;
    
    public boolean getPrediction() {
        return predicted;
    }        
        
}
