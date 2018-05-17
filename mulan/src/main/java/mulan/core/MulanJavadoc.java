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
package mulan.core;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.core.AllJavadoc;
import weka.core.Utils;

/**
 <!-- globalinfo-start -->
 * This class uses weka's Javadoc auto-generation classes to generate Javadoc<br>
 * comments and replaces the content between certain comment tags.
 * <br>
 <!-- globalinfo-end -->
 * 
 * @author Eleftherios Spyromitros-Xioufis
 * @author Grigorios Tsoumakas
 * @version 2012.02.27
 */
public class MulanJavadoc {

    static File originalDir;
    static String dir;

    /**
     * Sets the direcrory
     *
     * @param dir the directory path
     */
    public static void setDir(String dir) {
        MulanJavadoc.dir = dir;
    }

    /**
     * Command line interface
     *
     * @param args command line args
     * @throws java.lang.Exception Potential exception thrown. To be handled in an upper level.
     */
    public static void main(String[] args) throws Exception {
        dir = Utils.getOption("dir", args);
        originalDir = new File(dir);
        traverse();
    }

    /**
     * Updates comments
     *
     * @param classname the name of the class
     */
    public static void updateJavadoc(String classname) {
        try {
            AllJavadoc jdoc = new AllJavadoc();
            jdoc.setClassname(classname);
            jdoc.setDir(dir);
            jdoc.setUseStars(false);

            String result = jdoc.updateJavadoc();

            File file;
            BufferedWriter writer;

            if (!result.isEmpty()) {
                file = new File(jdoc.getDir() + "/" + jdoc.getClassname().replaceAll("\\.", "/") + ".java");
                if (!file.exists()) {
                    System.out.println("File '" + file.getAbsolutePath() + "' doesn't exist!");
                    return;
                }

                writer = new BufferedWriter(new FileWriter(file));
                writer.write(result);
                writer.close();

                System.out.println(jdoc.getClassname() + "'s Javadoc successfully updated.");
            } else {
                System.out.println(jdoc.getClassname() + "'s Javadoc update failed. Skipping file..");
            }
        } catch (Exception ex) {
            Logger.getLogger(MulanJavadoc.class.getName()).log(Level.SEVERE, null, ex);
        }

    }

    /**
     * Recursively visit all files
     *
     * @throws java.lang.Exception if failed
     */
    public static void traverse() throws Exception {
        recursiveTraversal(originalDir);
    }

    /**
     * Recursively visit all files
     *
     * @param fileObject the path of the file or directory
     * @throws java.lang.Exception if failed
     */
    public static void recursiveTraversal(File fileObject) throws Exception {
        if (fileObject.isDirectory()) {
            //System.out.println(fileObject.getName());
            File allFiles[] = fileObject.listFiles();
            for (File aFile : allFiles) {
                recursiveTraversal(aFile);
            }
        } else if (fileObject.isFile()) {
            String classname = fileObject.getPath();
            if (classname.endsWith(".java")) {
                classname = classname.replace(".java", "");
                classname = classname.replace(dir, "");
                classname = classname.replaceAll("\\\\", "\\.");
                System.out.println(classname);
                updateJavadoc(classname);
            }
        }
    }

    /**
     * Returns global information about the class
     *
     * @return global information
     */
    public String globalInfo() {
        return "This class uses weka's Javadoc auto-generation classes to generate Javadoc\n" + "comments and replaces the content between certain comment tags.";
    }

}