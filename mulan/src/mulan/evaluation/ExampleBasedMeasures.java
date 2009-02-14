package mulan.evaluation;

import java.util.List;

import mulan.classifier.Bipartition;
import weka.core.Utils;


public class ExampleBasedMeasures {
	private double hammingLoss;
	private double subsetAccuracy;
	private double accuracy;
	private double recall;
	private double precision;
	private double fMeasure;

	private double forgivenessRate;

	protected ExampleBasedMeasures(List<ModelEvaluationDataPair<Bipartition>> predictionData, double forgivenessRate){
        this.forgivenessRate = forgivenessRate;
        computeMeasures(predictionData);
	}
	
	protected ExampleBasedMeasures(List<ModelEvaluationDataPair<Bipartition>> predictionData){
		this (predictionData, 1.0);
	}

    /*
     * Aggregation of array of measures
     */
    protected ExampleBasedMeasures(ExampleBasedMeasures[] arrayOfMeasures) {
		accuracy = 0;
		hammingLoss = 0;
		precision = 0;
		recall = 0;
		fMeasure = 0;
		subsetAccuracy = 0;

        for (ExampleBasedMeasures measures : arrayOfMeasures) {
            accuracy += measures.getAccuracy();
            hammingLoss += measures.getHammingLoss();
            precision += measures.getPrecision();
            recall += measures.getRecall();
            fMeasure += measures.getFMeasure();
            subsetAccuracy += measures.getSubsetAccuracy();
        }

        int arraySize = arrayOfMeasures.length;
        accuracy /= arraySize;
        hammingLoss /= arraySize;
        precision /= arraySize;
        recall /= arraySize;
        fMeasure /= arraySize;
        subsetAccuracy /= arraySize;
    }
	
	private void computeMeasures(List<ModelEvaluationDataPair<Bipartition>> predictionData){
		// Reset in case of multiple calls
		accuracy = 0;
		hammingLoss = 0;
		precision = 0;
		recall = 0;
		fMeasure = 0;
		subsetAccuracy = 0;

		int numLabels = predictionData.get(0).getTrueLabels().size();
        for (ModelEvaluationDataPair<Bipartition> pair : predictionData) 
		{
			// Counter variables
			double setUnion = 0; // |Y or Z|
			double setIntersection = 0; // |Y and Z|
			double labelPredicted = 0; // |Z|
			double labelActual = 0; // |Y|
			double symmetricDifference = 0; // |Y xor Z|
			boolean setsIdentical = true; // innocent until proven guilty

			//Do the counting
            List<Boolean> trueLabels = pair.getTrueLabels(); 
            List<Boolean> predictedLabels = pair.getModelOutput().getBipartition();
			for (int j = 0; j < numLabels; j++)
			{
				boolean actual = trueLabels.get(j);
				boolean predicted = predictedLabels.get(j);

				if (predicted != actual)
				{
					symmetricDifference++;
					setsIdentical = false;
				}

				if (actual) labelActual++;
				if (predicted) labelPredicted++;

				if (predicted && actual) setIntersection++;
				if (predicted || actual) setUnion++;
			}

			if (setsIdentical) subsetAccuracy++;

			if(labelActual + labelPredicted == 0)
			{
				accuracy  += 1;
				recall    += 1;
				precision += 1;
				fMeasure  += 1;
			}
			else
			{
				if (Utils.eq(forgivenessRate, 1.0)) accuracy += (setIntersection / setUnion);
				else accuracy += Math.pow(setIntersection / setUnion, forgivenessRate);

				if (labelPredicted > 0) precision += (setIntersection / labelPredicted);
				if (labelActual > 0)    recall += (setIntersection / labelActual);
				
			}
			hammingLoss += (symmetricDifference / numLabels);
		}

		// Set final values
		int numInstances = predictionData.size();
		hammingLoss /= numInstances;
		accuracy /= numInstances;
		precision /= numInstances;
		recall /= numInstances;
		subsetAccuracy /= numInstances;
		fMeasure = computeF1Measure(precision, recall);
    }

  	private double computeF1Measure(double precision, double recall)
	{
	    if (Utils.eq(precision + recall, 0))
            return 0;
	    else
            return (2 * precision * recall) / (precision + recall);
	}
	
	public double getHammingLoss(){
		return hammingLoss;
	}
	
	public double getSubsetAccuracy(){
		return subsetAccuracy;
	}
	
	public double getAccuracy(){
		return accuracy;
	}
	
	public double getFMeasure(){
		return fMeasure;
	}
	public double getPrecision(){
		return precision;
	}
	
	public double getRecall(){
		return recall;
	}

}
