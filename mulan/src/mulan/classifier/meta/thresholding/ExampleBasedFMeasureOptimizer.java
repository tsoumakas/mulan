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
package mulan.classifier.meta.thresholding;

import java.util.Arrays;
import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.meta.MultiLabelMetaLearner;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.data.MultiLabelInstances;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

/**
 * <p>This class takes the marginal probabilities estimated for each label by a 
 * multi-label learner and transforms them into a bipartition which is 
 * approximately optimal for example-based FMeasure. This optimizer assumes 
 * independence of the target variables (labels) and the optimal solution always 
 * contains the labels with the highest marginal probabilities. For more 
 * information, see <em> Lewis, David (1995) Evaluating and optimizing 
 * autonomous text classification systems. In: Proceedings of the 18th annual 
 * international ACM SIGIR conference on Research and development in information 
 * retrieval (SIGIR 1995).</em></p>
 *
 * @author Eleftherios Spyromitros-Xioufis
 * @version 2012.07.16
 */
public class ExampleBasedFMeasureOptimizer extends MultiLabelMetaLearner {

    /**
     * The supplied multi-label learner should be able to output marginal
     * probabilities.
     *
     * @param baseLearner the base MultilabelLearner used 
     */
    public ExampleBasedFMeasureOptimizer(MultiLabelLearner baseLearner) {
        super(baseLearner);
    }

    /**
     * Default constructor
     */
    public ExampleBasedFMeasureOptimizer() {
        this(new BinaryRelevance(new J48()));
    }

    @Override
    protected void buildInternal(MultiLabelInstances trainingSet) throws Exception {
        baseLearner.build(trainingSet);
    }

    @Override
    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception,
            InvalidDataException {
        MultiLabelOutput mlo = baseLearner.makePrediction(instance);
        double[] marginals = mlo.getConfidences();
        boolean[] bipartition = bipartitionFromMarginals(marginals);
        mlo = new MultiLabelOutput(bipartition, marginals);
        return mlo;
    }

    /**
     * This method takes as input the marginal probabilities for each label and
     * returns a bipartition approximately optimized for example-based F-Measure
     *
     * @param confidences
     * @return
     */
    private boolean[] bipartitionFromMarginals(double[] confidences) {
        boolean[] bipartition = new boolean[numLabels];

        int[] sortedIndices = weka.core.Utils.stableSort(confidences);
        double[] sortedConfidences = Arrays.copyOfRange(confidences, 0, confidences.length);
        Arrays.sort(sortedConfidences);
        double BestF = 0;
        double topN = 0;
        for (int i = 0; i < numLabels; i++) {
            double nominator = 0;
            double denom1 = 0;
            double denom2 = 0;
            for (int j = 0; j < numLabels; j++) {
                double h;
                if (j > i) {
                    h = 0;
                } else {
                    h = 1;
                }
                nominator += sortedConfidences[numLabels - 1 - j] * h;
                denom1 += sortedConfidences[numLabels - 1 - j];
                denom2 += h;
            }
            double F = (2 * nominator) / (denom1 + denom2);
            if (F > BestF) {
                BestF = F;
                topN++;
            }
        }

        if (topN == 0) {
            // always output at least one label (the one with the highest
            // marginal probability)
            topN = 1;
        }
        for (int i = 0; i < topN; i++) {
            bipartition[sortedIndices[numLabels - 1 - i]] = true;
        }
        return bipartition;
    }

    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result = new TechnicalInformation(Type.INPROCEEDINGS);
        result.setValue(Field.AUTHOR, "David Lewis");
        result.setValue(Field.TITLE, "Evaluating and optimizing autonomous text classification systems");
        result.setValue(Field.BOOKTITLE, "Proceedings of the 18th annual international ACM SIGIR conference on Research and development in information retrieval (SIGIR 1995)");
        result.setValue(Field.YEAR, "1995");
        return result;
    }

}