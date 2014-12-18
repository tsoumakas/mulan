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
package mulan.examples;

import mulan.data.Statistics;
import mulan.data.MultiLabelInstances;
import weka.core.Utils;

/**
 * Class that calculates multi-label stastistics for a multi-label dataset
 *
 * @author Grigorios Tsoumakas
 * @version 2012.02.06
 */
public class EstimationOfStatistics {

    /**
     * Executes this example
     *
     * @param args command-line arguments -path and -filestem, e.g.
     * -path datasets/ -filestem emotions
     * @throws Exception exceptions not caught
     */
    public static void main(String[] args) throws Exception {
        String path = Utils.getOption("path", args);
        String filestem = Utils.getOption("filestem", args);
        MultiLabelInstances mlData = new MultiLabelInstances(path + filestem + ".arff", path + filestem + ".xml");

        Statistics stats = new Statistics();
        stats.calculateStats(mlData);
        System.out.println(stats);
    }
}
