package mulan.classifier.lazy;


/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;

import mulan.classifier.AbstractMultiLabelClassifier;
import mulan.classifier.LabelPowerset;
import mulan.classifier.Prediction;
import mulan.classifier.RAKEL;
import mulan.evaluation.BinaryPrediction;
import mulan.evaluation.Evaluator;
import mulan.evaluation.IntegratedEvaluation;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.core.TechnicalInformation;
import weka.core.Utils;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 * Class that implements the RAKEL (Random k-labelsets) algorithm <p>
 *
 * @author Grigorios Tsoumakas 
 * @version $Revision: 0.04 $ 
 */
@SuppressWarnings("serial")
public class RAKELmultiknn extends AbstractMultiLabelClassifier
{
    /**
     * Seed for replication of random experiments
     */
    private int seed=0;
            
    /**
     * Random number generator
     */
    private Random rnd;	

    /**
     * If true then the confidence of the base classifier to the decisions...
     */
    private boolean useConfidences = true;
    
    double[][] sumVotesIncremental; /* comment */
    double[][] lengthVotesIncremental;
    double[] sumVotes;
    double[] lengthVotes;
    int numOfModels;
    int sizeOfSubset;
    int[][] classIndicesPerSubset;
    int[][] absoluteIndicesToRemove;
    MultiKnn[] subsetClassifiers;
    protected Instances[] metadataTest;
    HashSet<String> combinations;		
    BinaryPrediction[][] predictions;
    boolean incremental =true;
    boolean cvParamSelection=false;
    int cvNumFolds, cvMinK, cvMaxK, cvStepK, cvMaxM, cvThresholdSteps;
    double cvThresholdStart, cvThresholdIncrement;    
        
  
               
    public RAKELmultiknn(int labels) {
        super(labels);
    }
    
    public void setSeed(int x) {
        seed = x;
        rnd = new Random(seed);
    }
    
    public RAKELmultiknn(int labels, int models, int subset) {
        this(labels);
        numOfModels = models;
        sizeOfSubset = subset;
        classIndicesPerSubset = new int[numOfModels][sizeOfSubset];
        absoluteIndicesToRemove = new int[numOfModels][sizeOfSubset];
        subsetClassifiers = new MultiKnn[numOfModels];
        metadataTest = new Instances[numOfModels];
    }
	
	public void setSizeOfSubset(int size) {
		sizeOfSubset=size;
		classIndicesPerSubset = new int[numOfModels][sizeOfSubset];		
		absoluteIndicesToRemove = new int[numOfModels][sizeOfSubset];
	}
	
        public int getSizeOfSubset() {
            return sizeOfSubset;
        }
        
	public void setNumModels(int models) {
		numOfModels = models;
		classIndicesPerSubset = new int[numOfModels][sizeOfSubset];
		absoluteIndicesToRemove = new int[numOfModels][sizeOfSubset];
		subsetClassifiers = new MultiKnn[numOfModels];
		metadataTest = new Instances[numOfModels];
	}
	
        public int getNumModels() {
            return numOfModels;
        }
                
        
	public BinaryPrediction[][] getPredictions() {
		return predictions;
	}
		
        private int binomial(int n, int m) 
        {
            int[] b = new int[n+1];
            b[0]=1;
            for (int i=1; i<=n; i++)
            {
                b[i] = 1;
                for (int j=i-1; j>0; --j)
                    b[j] += b[j-1];
            }
            return b[m];
        }
        
 
        public void setParamSets(int numFolds, int minK, int maxK, int stepK, int maxM, double thresholdStart, double thresholdIncrement, int thresholdSteps){
            cvNumFolds = numFolds;
            cvMinK = minK;
            cvMaxK = Math.min(numLabels-1, maxK);
            cvStepK = stepK;
            cvMaxM = maxM;
            cvThresholdStart = thresholdStart;
            cvThresholdIncrement = thresholdIncrement;
            cvThresholdSteps = thresholdSteps;
        }
        
        public void setParamSelectionViaCV(boolean flag){
            cvParamSelection = flag;
        }

	/**
         * This function evaluates different parameter sets for RAKEL, based 
         * on the values given by setParamSets. It then selects the best of 
         * these parameter sets based on Fmeasure. 
         * 
	 * @param trainData:
	 *            the data that will be used for parameter selection
	 */         
        public void paramSelectionViaCV(Instances trainData) throws Exception {                       
            ArrayList []metric = new ArrayList[cvNumFolds];
            //* Evaluate using X-fold CV
            for (int f=0; f<cvNumFolds; f++)
            {         
                metric[f] = new ArrayList();
                Instances foldTrainData = trainData.trainCV(cvNumFolds, f);
                Instances foldTestData = trainData.testCV(cvNumFolds, f);
            
                // rakel    
                for (int k=cvMinK; k<=cvMaxK; k+=cvStepK)
                {            
                    RAKEL rakel = new RAKEL(numLabels,binomial(numLabels, k), k);
                    rakel.setSeed(seed);
                    rakel.setBaseClassifier(baseClassifier);
                    int finalM = Math.min(binomial(numLabels,k),cvMaxM);
                    for (int m=0; m<finalM; m++)
                    {
                        rakel.updateClassifier(foldTrainData, m);
                        Evaluator evaluator = new Evaluator();
                        rakel.updatePredictions(foldTestData, m);
                        rakel.nullSubsetClassifier(m);
                        IntegratedEvaluation[] results = evaluator.evaluateOverThreshold(rakel.getPredictions(), foldTestData, cvThresholdStart, cvThresholdIncrement, cvThresholdSteps);                      
                        for (int t=0; t<results.length; t++)  {
                            metric[f].add(results[t].microFmeasure());                                                        
                        }
                    }
                }
            }
            ArrayList finalResults = new ArrayList();
            for (int i=0; i<metric[0].size(); i++) {                
                double sum=0;
                for (int f=0; f<cvNumFolds; f++)
                    sum = sum + (Double) metric[f].get(i);
                finalResults.add(sum/cvNumFolds);
            }
            
            double bestMetric=0; 
            int bestK=cvMinK, bestM=0;
            double bestT=cvThresholdStart;
            int counter=0;
            for (int k=cvMinK; k<=cvMaxK; k+=cvStepK)
            {            
                int finalM = Math.min(binomial(numLabels,k),cvMaxM);
                for (int m=0; m<finalM; m++)
                {
                    for (int t=0; t<cvThresholdSteps; t++)  {
                        double avgMetric=0;
                        for (int f=0; f<cvNumFolds; f++)
                            avgMetric += (Double) metric[f].get(counter);
                        avgMetric /= cvNumFolds;
                        if (avgMetric > bestMetric) {
                            bestK = k;
                            bestM = m;
                            bestT = cvThresholdStart+cvThresholdIncrement*t;
                            bestMetric = avgMetric;
                        }
                        counter++;
                    }
                }
            }
            /*
            System.out.println("Selected Parameters\n" +
                               "Subset size     : " + bestK + 
                               "Number of models: " + bestM +
                               "Threshold       : " + bestT);
            */
            setSizeOfSubset(bestK);
            setNumModels(bestM);
            setThreshold(bestT);            
        }        
        

        
        public void updatePredictions(Instances testData, int model) throws Exception {
		if (predictions == null) {
			predictions = new BinaryPrediction[testData.numInstances()][numLabels];
			sumVotesIncremental = new double[testData.numInstances()][numLabels];
			lengthVotesIncremental = new double[testData.numInstances()][numLabels];
		}
		
		for(int i = 0; i < testData.numInstances(); i++)
		{
			Instance instance = testData.instance(i);
			Prediction result = updatePrediction(instance, i, model);
//			Prediction result = makePrediction(instance);
			//System.out.println(java.util.Arrays.toString(result.getConfidences()));
			for(int j = 0; j < numLabels; j++)
			{
				int classIdx = testData.numAttributes() - numLabels + j;
				boolean actual = Utils.eq(1, instance.value(classIdx));
				predictions[i][j] = new BinaryPrediction(
							result.getPrediction(j), 
							actual, 
							result.getConfidence(j));
			}
		}		
	}
	
        
        
    @Override
    public void buildClassifier(Instances trainData) throws Exception {
        calcIndexes(trainData);
        
        if (cvParamSelection) 
            paramSelectionViaCV(trainData);

        // need a structure to hold different combinations
        combinations = new HashSet<String>();		

        for (int i=0; i<numOfModels; i++)
            updateClassifier(trainData, i);		
    }
	
    public void updateClassifier(Instances trainData, int model) throws Exception {
        if (indexOfTrue == null)
            calcIndexes(trainData);

        if (combinations == null)
            combinations = new HashSet<String>();

        // select a random subset of classes not seen before
        // todo: select according to invere distribution of current selection
        boolean[] selected;
        do {
            selected = new boolean[numLabels];
            for (int j=0; j<sizeOfSubset; j++) {
                int randomLabel;
                randomLabel = Math.abs(rnd.nextInt() % numLabels);
                while (selected[randomLabel] != false) {
                    randomLabel = Math.abs(rnd.nextInt() % numLabels);
                }
                selected[randomLabel] = true;
                //System.out.println("label: " + randomLabel);
                classIndicesPerSubset[model][j] = randomLabel;
            }
            Arrays.sort(classIndicesPerSubset[model]);
        } while (combinations.add(Arrays.toString(classIndicesPerSubset[model])) == false);
        System.out.println("Building model " + model + ", subset: " + Arrays.toString(classIndicesPerSubset[model]));	

        // remove the unselected labels
        int numPredictors = trainData.numAttributes()-numLabels;
        absoluteIndicesToRemove[model] = new int[numLabels-sizeOfSubset]; 
        int k=0;
        for (int j=0; j<numLabels; j++) 
            if (selected[j] == false) {
                absoluteIndicesToRemove[model][k] = numPredictors+j;
                k++;					
            }				                     
        Remove remove = new Remove();
        remove.setAttributeIndicesArray(absoluteIndicesToRemove[model]);
        remove.setInputFormat(trainData);
        remove.setInvertSelection(false);
        Instances trainSubset = Filter.useFilter(trainData, remove);
        //System.out.println(trainSubset.toSummaryString());
        //System.out.println(trainSubset.numInstances());

        // build a LabelPowersetClassifier for the selected label subset;
        subsetClassifiers[model] = new MultiKnn(6, 10);
        subsetClassifiers[model].buildClassifier(trainSubset);

        // keep the header of the training data for testing
        trainSubset.delete();
        metadataTest[model] = trainSubset;
	}
	                
	public Prediction updatePrediction(Instance instance, int instanceNumber, int model) throws Exception {	
            int numPredictors = instance.numAttributes()-numLabels;

            // transform instance
            //// new2 solution

            Instance newInstance;
            if (instance instanceof SparseInstance) {
                    newInstance = new SparseInstance(instance);
                    for (int i=1; i<numLabels-sizeOfSubset; i++)
                            newInstance.deleteAttributeAt(newInstance.numAttributes());
            } else {
                    double[] vals = new double[numPredictors+sizeOfSubset];
                    for (int j=0; j<vals.length-sizeOfSubset; j++)
                            vals[j] = instance.value(j);
                    newInstance = new Instance(instance.weight(), vals);			
            }


            //// new solution
            /*
            double[] vals = new double[numPredictors+sizeOfSubset];
            for (int j=0; j<vals.length-sizeOfSubset; j++)
                    vals[j] = instance.value(j);
            Instance newInstance = (instance instanceof SparseInstance)
            ? new SparseInstance(instance.weight(), vals)
            : new Instance(instance.weight(), vals);
            */

            //// old solution
            /*
            Instance newInstance = new Instance(numPredictors+sizeOfSubset);
            for (int j=0; j<newInstance.numAttributes(); j++)
                    newInstance.setValue(j, instance.value(j));
            */

            newInstance.setDataset(metadataTest[model]);

            double[] predictions = subsetClassifiers[model].makePrediction(newInstance).getPredictedLabels();
            for (int j=0; j<sizeOfSubset; j++) {
                sumVotesIncremental[instanceNumber][classIndicesPerSubset[model][j]] += predictions[j];
                lengthVotesIncremental[instanceNumber][classIndicesPerSubset[model][j]]++;
            }
            /*
            for (int i=0; i<numLabels; i++)
                    System.out.print(instance.value(numPredictors+i) + " ");
            System.out.println("");
            System.out.println(Arrays.toString(sumVotesIncremental[instanceNumber]));
            System.out.println(Arrays.toString(lengthVotesIncremental[instanceNumber]));
            //*/

            double[] confidence = new double[numLabels];
            double[] labels = new double[numLabels];
            for (int i=0; i<numLabels; i++) {
                    confidence[i] = sumVotesIncremental[instanceNumber][i]/lengthVotesIncremental[instanceNumber][i];
                    if (confidence[i] >= 0.5)
                            labels[i] = 1;
                    else
                            labels[i] = 0;
            }

            Prediction pred = new Prediction(labels, confidence);

            return pred;
	}
	
	
	public Prediction makePrediction(Instance instance) throws Exception {		
            int numPredictors = instance.numAttributes()-numLabels;
            double[] sumConf = new double[numLabels];
            sumVotes = new double[numLabels];
            lengthVotes = new double[numLabels];

            // transform instance
            // note: should be placed inside the for if different subset sizes
            double[] vals = new double[numPredictors+sizeOfSubset];
            for (int j=0; j<vals.length-sizeOfSubset; j++)
                    vals[j] = instance.value(j);
            Instance newInstance = (instance instanceof SparseInstance)
            ? new SparseInstance(instance.weight(), vals)
            : new Instance(instance.weight(), vals);
            
            for (int i=0; i<numOfModels; i++) {
                // what for is this if?
                //if (subsetClassifiers[i] == null)
                //    continue;

                newInstance.setDataset(metadataTest[i]);
                //System.out.println("new instance: " + newInstance.toString());

                Prediction pred = subsetClassifiers[i].makePrediction(newInstance);
                
                for (int j=0; j<sizeOfSubset; j++) {
                    sumConf[classIndicesPerSubset[i][j]] += pred.getConfidence(j);                   
                    sumVotes[classIndicesPerSubset[i][j]] += (pred.getPrediction(j)) ? 1 : 0;
                    lengthVotes[classIndicesPerSubset[i][j]]++;
                }
            }
            /*
            for (int i=0; i<numLabels; i++)
                    System.out.print(instance.value(numPredictors+i) + " ");
            System.out.println("");
            System.out.println(Arrays.toString(sumVotes));
            System.out.println(Arrays.toString(lengthVotes));
            //*/

            double[] confidence = new double[numLabels];
            double[] labels = new double[numLabels];
            for (int i=0; i<numLabels; i++) {
                if (lengthVotes[i] != 0)
                    confidence[i] = sumConf[i]/lengthVotes[i];
                    //confidence[i] = sumVotes[i]/lengthVotes[i];
                else
                    confidence[i] = 0;
                if (confidence[i] >= 0.5)
                    labels[i] = 1;
                else
                    labels[i] = 0;
            }

            Prediction pred = new Prediction(labels, confidence);

            return pred;
	}
        
        public void nullSubsetClassifier(int i) {
            subsetClassifiers[i] = null;
        }

    public String getRevision() {
        throw new UnsupportedOperationException("Not supported yet.");
    }
}
