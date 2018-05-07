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
package mulan.classifier.transformation;

import mulan.data.MultiLabelInstances;
import weka.classifiers.Classifier;

/**
 * <p>An algorithm that trains both binary and pairwise models.</p>
 *
 * @author Grigorios Tsoumakas
 * @version 2012.11.25
 */
public abstract class BinaryAndPairwise extends TransformationBasedMultiLabelLearner {

    /**
     * Binary models
     */
    private BinaryRelevance oneVsRestModels;
    /**
     * Pairwise models
     */
    private Pairwise oneVsOneModels;

    /**
     * Constructor that initializes the learner with a base algorithm
     *
     * @param classifier the binary classification algorithm to use
     */
    public BinaryAndPairwise(Classifier classifier) {
        super(classifier);
    }

    protected Pairwise getOneVsOneModels() {
        return oneVsOneModels;
    }

    protected BinaryRelevance getOneVsRestModels() {
        return oneVsRestModels;
    }

    @Override
    protected void buildInternal(MultiLabelInstances trainingSet) throws Exception {
        debug("Building binary (one-vs-rest) models");
        oneVsRestModels = new BinaryRelevance(baseClassifier);
        oneVsRestModels.setDebug(getDebug());
        oneVsRestModels.build(trainingSet);

        debug("Building pairwise (one-vs-one) models");
        oneVsOneModels = new Pairwise(baseClassifier);
        oneVsOneModels.setDebug(getDebug());
        oneVsOneModels.build(trainingSet);
    }
}