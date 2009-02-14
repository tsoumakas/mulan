package mulan.evaluation;

import java.util.List;

import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.core.FastVector;
import weka.core.Instances;
import weka.core.Utils;

public class ConfidenceLabelBasedMeasures {

    double[] auc = new double[2];
    double[] labelAUC;

    protected ConfidenceLabelBasedMeasures(List<ModelEvaluationDataPair<List<Double>>> predictions) {
       computeMeasures(predictions);
    }	
		
    private void computeMeasures(List<ModelEvaluationDataPair<List<Double>>> predictions) {
        int numLabels = predictions.get(0).getNumLabels();
        
        // AUC
        FastVector[] m_Predictions = new FastVector[numLabels];
		for (int j=0; j<numLabels; j++)
            m_Predictions[j] = new FastVector();
        FastVector all_Predictions = new FastVector();


        for (ModelEvaluationDataPair<List<Double>> pair : predictions)
		{
            for (int j = 0; j < numLabels; j++)
            {

                int classValue;
                boolean actual = pair.getTrueLabels().get(j);
                if (actual)
                    classValue = 1;
                else
                    classValue = 0;

                double[] dist = new double[2];
                dist[1] = pair.getModelOutput().get(j);
                dist[0] = 1 - dist[1];

                m_Predictions[j].addElement(new NominalPrediction(classValue, dist, 1));
                all_Predictions.addElement(new NominalPrediction(classValue, dist, 1));
            }
        }

        for (int i=0; i<numLabels; i++) {
            ThresholdCurve tc = new ThresholdCurve();
            Instances result = tc.getCurve(m_Predictions[i], 1);
            labelAUC[i] = ThresholdCurve.getROCArea(result);
		}
        auc[Averaging.MACRO.ordinal()] = Utils.mean(labelAUC);
        ThresholdCurve tc = new ThresholdCurve();
        Instances result = tc.getCurve(all_Predictions, 1);
        auc[Averaging.MICRO.ordinal()] = ThresholdCurve.getROCArea(result);
    }

    public double getLabelAUC(int label) {
        return labelAUC[label];
    }
	
	public double getAUC(Averaging averagingType){
		return auc[averagingType.ordinal()];
	}

}
