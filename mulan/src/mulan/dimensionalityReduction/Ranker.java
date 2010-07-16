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
 *    Ranker.java
 *    Copyright (C) 2009-2010 Aristotle University of Thessaloniki, Thessaloniki, Greece
 */
package mulan.dimensionalityReduction;

import mulan.data.MultiLabelInstances;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.AttributeEvaluator;
import weka.core.Instances;
import weka.core.Range;

/**
 * Ranks attributes according to an AttributeEvaluator. It internally uses Weka's
 * Ranker, initialized so as to neglect the labels.
 *
 * @author Grigorios Tsoumakas
 */
public class Ranker {

    /**
     * Calls a specified {@link AttributeEvaluator} to evaluate each feature attribute
     * of specified {@link MultiLabelInstances} data set.
     * Internally it uses {@link weka.attributeSelection.Ranker}, where
     * {@link weka.attributeSelection.Ranker#setStartSet(String)} is preset with range of
     * only feature attributes indices.
     *
     * @param attributeEval the attribute evaluator to guide the search
     * @param mlData the multi-label instances data set
     * @return an array (not necessarily ordered) of selected attribute indexes
     * @throws Exception if an error occur in search
     */
    public int[] search(AttributeEvaluator attributeEval, MultiLabelInstances mlData) throws Exception {
        Instances data = mlData.getDataSet();
        String startSet = Range.indicesToRangeList(mlData.getLabelIndices());
        weka.attributeSelection.Ranker wekaRanker = new weka.attributeSelection.Ranker();
        wekaRanker.setStartSet(startSet);
        return wekaRanker.search((ASEvaluation) attributeEval, data);
    }
}
