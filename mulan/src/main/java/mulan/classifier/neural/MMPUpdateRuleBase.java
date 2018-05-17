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
 *    MMPUpdateRuleBase.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package mulan.classifier.neural;

import java.util.List;
import java.util.Map;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.neural.model.Neuron;
import mulan.core.ArgumentNullException;
import mulan.evaluation.loss.RankingLossFunction;

/**
 * The base class of update rules for {@link MMPLearner}. The base class implements the
 * {@link ModelUpdateRule} interface and provides a common logic shared among update rules 
 * for {@link MMPLearner}. More information on uprate rules logic can be found in paper referenced
 * by {@link MMPLearner}.
 * 
 * @see MMPLearner 
 * @author Jozef Vilcek
 * @version 2012.02.27
 */
public abstract class MMPUpdateRuleBase implements ModelUpdateRule {

    /** The list of Neurons representing the model to be updated by the rule in learning process */
    private final List<Neuron> perceptrons;
    /** The masure used to decide when (and to what extend) the model should be updated by the rule */
    private final RankingLossFunction lossFunction;

    /**
     * Creates a new instance of {@link MMPUpdateRuleBase}.
     *
     * @param perceptrons the list of perceptrons, representing the model, which will receive updates.
     * @param loss the lossFunction measure used to decide when the model should be updated by the rule
     */
    public MMPUpdateRuleBase(List<Neuron> perceptrons, RankingLossFunction loss) {
        if (perceptrons == null) {
            throw new ArgumentNullException("perceptrons");
        }
        if (loss == null) {
            throw new ArgumentNullException("lossMeasure");
        }
        this.perceptrons = perceptrons;
        this.lossFunction = loss;
    }

    public final double process(DataPair example, Map<String, Object> params) {
        int numLabels = example.getOutput().length;
        int numFeatures = example.getInput().length;
        double[] dataInput = example.getInput();
        double[] confidences = new double[numLabels];

        // update model prediction on raking for given example
        for (int index = 0; index < numLabels; index++) {
            Neuron perceptron = perceptrons.get(index);
            confidences[index] = perceptron.processInput(dataInput);
        }
        MultiLabelOutput mlOut = new MultiLabelOutput(confidences);

        // get a lossFunction measure of a model for given example
        double loss = lossFunction.computeLoss(mlOut.getRanking(), example.getOutputBoolean());
        if (loss != 0) {
            // update update parameters for perceptrons
            double[] updateParams = computeUpdateParameters(example, confidences, loss);
            // perform updates of perceptrons
            for (int lIndex = 0; lIndex < numLabels; lIndex++) {
                Neuron perceptron = perceptrons.get(lIndex);
                double[] weights = perceptron.getWeights();
                for (int iIndex = 0; iIndex < numFeatures; iIndex++) {
                    weights[iIndex] += updateParams[lIndex] * dataInput[iIndex];
                }
                // update bias weight
                //weights[numFeatures] += updateParams[lIndex] * perceptron.getBiasInput();
            }
        }

        return loss;
    }

    /**
     * Computes update parameters for each perceptron which will be subsequently used
     * for updating the weights. The function is called internally from
     * {@link MMPUpdateRuleBase#process(DataPair, Map)} function, when update of model for
     * given input example is needed.
     *
     * @param example the input example
     * @param confidences the confidences outputed by the model the input example
     * @param loss the lossFunction measure of the model for given input example
     * @return the parameters for updating preceptrons
     */
    protected abstract double[] computeUpdateParameters(DataPair example,
            double[] confidences, double loss);
}