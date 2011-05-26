package mulan.classifier.transformation;

import java.util.Random;

import java.util.logging.Level;
import java.util.logging.Logger;
import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

/**
 * <!-- globalinfo-start -->
 *
 * <pre>
 * Class implementing the EPS algorithm, which constructs an ensemble of
 * Pruned Sets models, via sampling
 * </pre>
 *
 * <!-- globalinfo-end -->
 *
 * <!-- technical-bibtex-start --> BibTeX:
 * 
 * <!-- technical-bibtex-end -->
 *
 * @author Emmanouela Stachtiari
 * @author Grigorios Tsoumakas
 * @version June 6, 2010
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

    /**
     * Returns an instance of a TechnicalInformation object, containing detailed
     * information about the technical background of this class, e.g., paper
     * reference or book this class is based on.
     *
     * @return the technical information about this class
     */
    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result = new TechnicalInformation(Type.CONFERENCE);
        result.setValue(Field.AUTHOR, "Read, Jesse");
        result.setValue(Field.TITLE, "Multi-label Classification using Ensembles of Pruned Sets");
        result.setValue(Field.PAGES, "995-1000");
        result.setValue(Field.BOOKTITLE, "ICDM'08: Eighth IEEE International Conference on Data Mining");
        result.setValue(Field.YEAR, "2008");

        return result;
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
