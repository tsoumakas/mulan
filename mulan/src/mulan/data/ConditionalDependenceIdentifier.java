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

/*
 *    ConditionalDependenceIdentifier.java
 */
package mulan.data;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Remove;

/**
 * A class for identification of conditional dependence between each pair of labels. The conditional dependence between each pair of labels
 * is estimated by evaluating the advantage gained from exploiting this dependence for binary classification of each one of the labels.
 * Following the definition of conditional independence, for two conditionally independent labels, predictions of a label by probability-based classification
 * models trained once on a regular features space and second on the features space augmented by the second label should be at least very similar. For this
 * estimation two binary classifiers are trained and their accuracy is estimated using k-fold cross-validation. If the accuracy of the  model trained on the
 * features space augmented by the second label is significantly higher, the labels are considered conditionally dependent. The statistical significance of
 * the difference between both classifiers is determined using a paired t-test. This procedure is performed for all possible label pairs considering the label
 * order in the pair . Among the two pairs with the same labels, the pair with maximal t-statistic value is added to the resulting list of dependent pairs. Finally,
 *  the resultant label pairs are sorted according to their t-statistic value in descending order (i.e., from the most to the least dependent pairs).
 *
 * @author Lena Chekina (lenat@bgu.ac.il)
 * @version 30.11.2010
 */
public class ConditionalDependenceIdentifier implements LabelPairsDependenceIdentifier, Serializable {

    /** A default t-critical value, corresponds to significance level 0.01. Label pairs with dependence value below the critical are considered as independent.*/
    private double criticalValue = 3.25;
    /** A single-label classifier used to perform dependence test between labels. */
    private  Classifier baseLearner;
    /** Number of folds used for cross validation. */
    private int numFolds = 10;
    /** Seed for replication of random experiments*/
    protected int seed;
    /** A caching mechanism for reusing once constructed models. */
    private static HashMap<String, FilteredClassifier> existingModels=null;

    /**
     *  Initializes a single-label classifier used to perform dependence test between labels and a caching mechanism for reusing constructed models.
     * @param classifier - a single-label classifier used to perform dependence test between labels.
     */
    public ConditionalDependenceIdentifier(Classifier classifier) {
        baseLearner = classifier;
        if (existingModels==null){
            existingModels = new HashMap<String,FilteredClassifier>();
        }
    }

    /**
     *  Calculates t-statistic value for each pair of labels.
     *
     * @param mlInstances the {@link mulan.data.MultiLabelInstances} dataset on which dependencies should be calculated
     * @return an array of label pairs sorted in descending order of the t-statistic value
     */
    public LabelsPair[] calculateDependence(MultiLabelInstances mlInstances){
        int numLabels = mlInstances.getNumLabels();
        int numPairs = numLabels*(numLabels-1)/2;
        LabelsPair[] pairs = new  LabelsPair[numPairs];
        int ind=0;
        for(int i=0; i<numLabels-1; i++){                                                                                                     //for each pair of labels i and j
            for(int j=i+1; j<numLabels;j++){
                int[] comb1 = new int[2];                                                                                                                 //will store a pair [i,j]
                int[] comb2 = new int[2];                                                                                                                 //will store a pair [j,i]
                comb1[0] = i;
                comb1[1] = j;
                comb2[0] = j;
                comb2[1] = i;
                double val1 = testDependence(comb1, mlInstances, numFolds);                                   // dependency test for classes  i  and  j
                double val2 = testDependence(comb2, mlInstances, numFolds);                                  // dependency test for classes  j  and  i
                if(val1>=val2){                                                                                                                               //add a pair with MAXIMAL value to the results object
                    pairs[ind++] = new LabelsPair(comb1, val1);
                }
                else{
                    pairs[ind++] = new LabelsPair(comb2, val2);
                }
            }
        }
        Arrays.sort(pairs, Collections.reverseOrder());
        return pairs;
    }

    /**
     *  Performs dependency test between two labels.
     *
     * @param comb an array with indexes of the two labels for the test
     * @param mlData the {@link mulan.data.MultiLabelInstances} dataset on which dependencies should be calculated
     * @param numFolds number of folds used for cross validation
     * @return a value indicating the level of dependence between two labels. As higher is value as more conditionally dependent are the labels. For independent labels "-1" is returned.
     */
    private double testDependence(int[] comb, MultiLabelInstances mlData, int numFolds) {
        double[] acc1 = null;
        double[] acc2 = null;
        double val;
        try{
            int numLabels = mlData.getNumLabels();
            int[] indecesToRemove1;
            int[] indecesToRemove2;
            int classIndex;
            final int[] labelIndices = mlData.getLabelIndices();
            Instances[] trainSets = new Instances[numFolds];
            Instances[] testSets = new Instances[numFolds];
            weka.classifiers.Evaluation[] eval = new weka.classifiers.Evaluation[numFolds];
            weka.classifiers.Evaluation[] eval2 = new weka.classifiers.Evaluation[numFolds];
            acc1 = new double[numFolds];
            acc2 = new double[numFolds];
            Instances workingSet = new Instances(mlData.getDataSet());
            Random random = new Random(seed);
            workingSet.randomize(random);                                                                                                     //randomize numFolds train-test pairs
            for (int i=0; i<numFolds; i++)                                                                                                       //build dependent and independent models on each fold
            {
                trainSets[i] = workingSet.trainCV(numFolds, i, random);
                testSets[i]  = workingSet.testCV(numFolds, i);
                classIndex = labelIndices[comb[0]];
                indecesToRemove1 = new int[numLabels-1];                                                                            //prepare indexes to build independent  model   (x -> L1)
                int counter2 = 0;
                for (int counter1 = 0; counter1<numLabels; counter1++){
                    if(counter1!=comb[0]){
                        indecesToRemove1[counter2] = labelIndices[counter1];
                        counter2++;
                    }
                }
                FilteredClassifier indepModel;
                int foldHash = trainSets[i].toString().hashCode();
                String modelKey = createKey(indecesToRemove1, foldHash);
                if (existingModels.containsKey(modelKey))  {
                    indepModel=existingModels.get(modelKey);                                                                   //Retrieving model from cache
                }
                else{
                    indepModel = buildModel(indecesToRemove1,classIndex, trainSets[i]);            //Building independent model for L1
                }
                indecesToRemove2 = new int[numLabels-2];                                                                           //prepare indexes to build dependent model  (x, L2 - > L1)
                counter2 = 0;
                for (int counter1 = 0; counter1<numLabels; counter1++){
                    if((counter1!=comb[0]) && (counter1!=comb[1])){
                        indecesToRemove2[counter2] = labelIndices[counter1];
                        counter2++;
                    }
                }
                FilteredClassifier depModel = buildModel(indecesToRemove2,                              //Building depend model for the L1 label
                        classIndex, trainSets[i]);

                //evaluate independent model
                Instances filteredTrainData = prepareDatSet(indecesToRemove1,classIndex,trainSets[i]);
                Instances filteredTestData = prepareDatSet(indecesToRemove1,classIndex,testSets[i]);
                eval[i] = new weka.classifiers.Evaluation(filteredTrainData);
                eval[i].evaluateModel(indepModel, filteredTestData);
                acc1[i] = eval[i]. pctCorrect();

                //evaluate  dependent model
                Instances filteredTrainData2 = prepareDatSet(indecesToRemove2,classIndex,trainSets[i]);
                Instances filteredTestData2 = prepareDatSet(indecesToRemove2,classIndex,testSets[i]);
                eval2[i] = new weka.classifiers.Evaluation(filteredTrainData2);
                eval2[i].evaluateModel(depModel, filteredTestData2);
                acc2[i] = eval2[i]. pctCorrect();
            }
        } catch (Exception e) {
            Logger.getLogger(ConditionalDependenceIdentifier.class.getSimpleName()).log(Level.SEVERE, null, e);
        }
        finally{
            if(acc1==null || acc2==null){
                val = -1;
            }
            else{
                val = applyTtest(acc1,acc2);                                                                                                       // /t-test on evaluation results
            }
        }
        return val;
    }

    /**
     * Performs paired t-test with same variances.
     *
     * @param val1 an array holding accuracy values of model1
     * @param val2  an array holding accuracy values of model2
     * @return  t-statistic representing result of the t-test applied on the arrays values. Return "-1" if average accuracy of model1 is higher than that of model2
     */
    private double applyTtest(double[] val1, double[] val2) {
        double sum1=0;
        double sum2=0;
        final int count = val1.length;

        //compute Average
        for (int i=0; i< count; i++) {
            sum1+=val1[i];
            sum2+=val2[i];
        }
        double avg1=sum1/count;
        double avg2=sum2/count;
        if(avg1>avg2){                                                                                                                                              // If average accuracy of independent model is higher than average accuracy
            return -1;                                                                                                                                                  //  of conditionally dependent model -> no need to model dependence!
        }

        //compute Variance
        double var1;
        double var2;
        double varDiff=0;
        for (int i=0; i< count; i++) {
            var1=val1[i]-avg1;
            var2=val2[i]-avg2;
            varDiff+=Math.pow(var1-var2,2);
        }

        //apply t-test
        double m =0;
        if(varDiff!=0){
            m= Math.sqrt(count*(count-1) / varDiff);
        }
        double tValue = (avg1 - avg2) * m;
        if (tValue < 0){
            tValue = tValue * (-1);
        }
        return tValue;
    }

    /**
     * Creating classification model.
     *
     * @param indicesToRemove indexes of labels to be removed from dataset
     * @param classIndex index of the label tested as class
     * @param trainDataset the {@link weka.core.Instances} dataset on which the model should be learned
     * @return {@link weka.classifiers.meta.FilteredClassifier} classification model
     * @throws Exception if creating the classification model fails
     */
    private FilteredClassifier buildModel(int[] indicesToRemove, int classIndex, Instances trainDataset) throws Exception {
        FilteredClassifier model = new FilteredClassifier();
        model.setClassifier( AbstractClassifier.makeCopy(baseLearner));
        Remove remove = new Remove();
        remove.setAttributeIndicesArray(indicesToRemove);
        remove.setInputFormat(trainDataset);
        remove.setInvertSelection(false);
        model.setFilter(remove);
        trainDataset.setClassIndex(classIndex);
        model.buildClassifier(trainDataset);
        int foldHash = trainDataset.toString().hashCode();
        String modelKey = createKey(indicesToRemove, foldHash);
        existingModels.put(modelKey, model);
        return model;
    }

    /**
     * Concatenate all integers from an array with additional integer into a single string.
     *
     * @param set an array representing labels subset
     * @param fold a hash code of the current training set
     * @return a string in the form: "_l1_l2_ ... ln_fold"
     */
    private String createKey(int[] set, int fold) {
        StringBuilder sb = new StringBuilder("_");
        for (int i : set){
            sb.append(i);
            sb.append("_");
        }
        sb.append(fold);
        return  sb.toString();
    }

    /**
     * Removes the specified features from the supplied dataset, and set the specified feature as class.
     *
     * @param indicesToRemove indexes of labels to be removed from the initial dataset
     * @param classIndex index of the class label
     * @param dataset the initial {@link weka.core.Instances} dataset
     * @return {@link weka.core.Instances} filtered dataset with set classIndex
     * @throws Exception if removal has failed
     */
    private Instances prepareDatSet(int[] indicesToRemove, int classIndex, Instances dataset) throws Exception {
        Remove remove = new Remove();
        remove.setAttributeIndicesArray(indicesToRemove);
        remove.setInputFormat(dataset);
        remove.setInvertSelection(false);
        dataset.setClassIndex(classIndex);
        return dataset;
    }

    /**
     *
     * @param criticalValue a t-critical value
     */
    public void setCriticalValue(double criticalValue) {
        this.criticalValue = criticalValue;
    }

    public double getCriticalValue() {
        return criticalValue;
    }

    /**
     *
     * @return The seed for replication of random experiments
     */
    public int getSeed() {
        return seed;
    }

    /**
     *
     * @param seed the seed for random generation
     */
    public void setSeed(int seed) {
        this.seed = seed;
    }

    /**
     *
     * @return Number of folds
     */
    public int getNumFolds() {
        return numFolds;
    }

    /**
     *
     * @param numFolds the number of folds
     */
    public void setNumFolds(int numFolds) {
        this.numFolds = numFolds;
    }
}
