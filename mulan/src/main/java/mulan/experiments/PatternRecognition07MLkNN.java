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
package mulan.experiments;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import mulan.classifier.lazy.MLkNN;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.measure.*;
import weka.core.Utils;

/**
 * <p>Class replicating the experiment in <em>Min-Ling Zhang and Zhi-Hua Zhou.
 * 2007. ML-KNN: A lazy learning approach to multi-label learning. Pattern
 * Recogn. 40, 7 (July 2007), 2038-2048. DOI=10.1016/j.patcog.2006.12.019</em>
 * </p>
 *
 * @author Eleftherios Spyromitros-Xioufis (espyromi@csd.auth.gr)
 * @version 2010.12.10
 */
public class PatternRecognition07MLkNN {

    /**
     * Main code
     *
     * @param args command line arguments
     */
    public static void main(String[] args) {
        try {

            String path = Utils.getOption("path", args);
            String filestem = Utils.getOption("filestem", args);

            System.out.println("Loading the data set");
            MultiLabelInstances dataSet = new MultiLabelInstances(path + filestem + ".arff", path + filestem + ".xml");

            Evaluator eval = new Evaluator();
            MultipleEvaluation results;
            List<Measure> measures = new ArrayList<>(5);
            measures.add(new HammingLoss());
            measures.add(new OneError());
            measures.add(new Coverage());
            measures.add(new RankingLoss());
            measures.add(new AveragePrecision());

            int numOfNeighbors;
            for (int i = 8; i <= 12; i++) {
                System.out.println("MLkNN Experiment for " + i + " neighbors:");
                numOfNeighbors = i;
                double smooth = 1.0;
                MLkNN mlknn = new MLkNN(numOfNeighbors, smooth);
                // mlknn.setDebug(true);
                results = eval.crossValidate(mlknn, dataSet, measures, 10);
                System.out.println(results);
            }
        } catch (Exception ex) {
            Logger.getLogger(PatternRecognition07MLkNN.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}