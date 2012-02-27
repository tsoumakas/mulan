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
 *    EnsembleOfPrunedSets.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package mulan.classifier.transformation;

import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

/**
 <!-- globalinfo-start -->
 * Class implementing the Ensemble of Pruned Sets algorithm(EPS) . For more information, see<br/>
 * <br/>
 * Read, Jesse, Pfahringer, Bernhard, Holmes, Geoff: Multi-label Classification using Ensembles of Pruned Sets. In: ICDM'08: Eighth IEEE International Conference on Data Mining, 995-1000, 2008.
 * <p/>
 <!-- globalinfo-end -->
 *
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;conference{Read2008,
 *    author = {Read, Jesse and Pfahringer, Bernhard and Holmes, Geoff},
 *    booktitle = {ICDM'08: Eighth IEEE International Conference on Data Mining},
 *    pages = {995-1000},
 *    title = {Multi-label Classification using Ensembles of Pruned Sets},
 *    year = {2008}
 * }
 * </pre>
 * <p/>
 <!-- technical-bibtex-end -->
 *
 * @author Emmanouela Stachtiari
 * @author Grigorios Tsoumakas
 * @version 2012.02.27
 */
public class EnsembleOfPrunedSets extends TransformationBasedMultiLabelLearner {

    /** Parameter for the threshold of discretization of prediction output */
    protected double threshold;
    /** Parameter for the number of models that constitute the ensemble*/
    protected int numOfModels;
    /** Percentage of data */
    protected double percentage;
    /** The models in the ensemble */
    protected PrunedSets[] ensemble;
    /** Random number generator */
    protected Random rand;

    /**
     * Creates a new instance with default values
     */
    public EnsembleOfPrunedSets() {
        this(66,10,0.5,2,PrunedSets.Strategy.A,3,new J48());
    }
    
    /**
     * @param aNumOfModels the number of models in the ensemble
     * @param aStrategy pruned sets strategy
     * @param aPercentage percentage of data to sample
     * @param aP pruned sets parameter p
     * @param aB pruned sets parameter b
     * @param baselearner the base learner
     * @param aThreshold the threshold for producing bipartitions
     */
    public EnsembleOfPrunedSets(double aPercentage, int aNumOfModels, double aThreshold, int aP, PrunedSets.Strategy aStrategy, int aB, Classifier baselearner) {
        super(baselearner);
        numOfModels = aNumOfModels;
        threshold = aThreshold;
        percentage = aPercentage;
        ensemble = new PrunedSets[numOfModels];
        for (int i = 0; i < numOfModels; i++) {
            try {
                ensemble[i] = new PrunedSets(AbstractClassifier.makeCopy(baselearner), aP, aStrategy, aB);
            } catch (Exception ex) {
                Logger.getLogger(EnsembleOfPrunedSets.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        rand = new Random(1);
    }

    @Override
    protected void buildInternal(MultiLabelInstances trainingSet)
            throws Exception {
        Instances dataSet = new Instances(trainingSet.getDataSet());

        for (int i = 0; i < numOfModels; i++) {
            dataSet.randomize(rand);
            RemovePercentage rmvp = new RemovePercentage();
            rmvp.setInputFormat(dataSet);
            rmvp.setPercentage(percentage);
            rmvp.setInvertSelection(true);
            Instances trainDataSet = Filter.useFilter(dataSet, rmvp);
            MultiLabelInstances train = new MultiLabelInstances(trainDataSet, trainingSet.getLabelsMetaData());
            ensemble[i].build(train);
        }
    }

    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result = new TechnicalInformation(Type.CONFERENCE);
        result.setValue(Field.AUTHOR, "Read, Jesse and Pfahringer, Bernhard and Holmes, Geoff");
        result.setValue(Field.TITLE, "Multi-label Classification using Ensembles of Pruned Sets");
        result.setValue(Field.PAGES, "995-1000");
        result.setValue(Field.BOOKTITLE, "ICDM'08: Eighth IEEE International Conference on Data Mining");
        result.setValue(Field.YEAR, "2008");

        return result;
    }

   /**
     * Returns a string describing classifier
     * @return a description suitable for displaying 
     */
    public String globalInfo() {

        return "Class implementing the Ensemble of Pruned Sets algorithm"
                + "(EPS) . For more information, see\n\n"
                + getTechnicalInformation().toString();
    }
    
    
    @Override
    protected MultiLabelOutput makePredictionInternal(Instance instance)
            throws Exception, InvalidDataException {

        int[] sumVotes = new int[numLabels];

        for (int i = 0; i < numOfModels; i++) {
            MultiLabelOutput ensembleMLO = ensemble[i].makePrediction(instance);
            boolean[] bip = ensembleMLO.getBipartition();

            for (int j = 0; j < sumVotes.length; j++) {
                sumVotes[j] += bip[j] == true ? 1 : 0;
            }
        }
        double[] confidence = new double[numLabels];

        for (int j = 0; j < sumVotes.length; j++) {
            confidence[j] = (double) sumVotes[j] / (double) numOfModels;
        }

        MultiLabelOutput mlo = new MultiLabelOutput(confidence, threshold);
        return mlo;
    }
}