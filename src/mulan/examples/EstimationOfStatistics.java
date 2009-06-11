/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package mulan.examples;

import mulan.core.Statistics;
import mulan.core.data.MultiLabelInstances;
import weka.core.Utils;

/**
 *
 * @author greg
 */
public class EstimationOfStatistics {

    public static void main(String[] args) throws Exception {
        String path = Utils.getOption("path", args);
        String filestem = Utils.getOption("filestem", args);
        MultiLabelInstances mlData = new MultiLabelInstances(path + filestem + ".arff", path + filestem + ".xml");

        Statistics stats = new Statistics();
        stats.calculateStats(mlData);
        System.out.println(stats);
    }

}
