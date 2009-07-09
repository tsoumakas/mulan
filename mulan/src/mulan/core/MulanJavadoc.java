package mulan.core;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Enumeration;
import java.util.Vector;

import weka.core.AllJavadoc;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Utils;

/**
 * <!-- globalinfo-start -->
 * This class uses weka's Javadoc auto-generation classes to generate Javadoc<br/>
 * comments and replaces the content between certain comment tags.
 * <p/>
 * <!-- globalinfo-end -->
 * 
 * <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -dir &lt;dir&gt;
 *  The directory where the mulan package resides.</pre>
 * 
 * <!-- options-end -->
 * 
 * @author Eleftherios Spyromitros-Xioufis ( espyromi@csd.auth.gr )
 * 
 */
public class MulanJavadoc implements OptionHandler {

	static File originalDir;
	static String dir;

	public static void main(String[] args) throws Exception {

		dir = Utils.getOption("dir", args);
		originalDir = new File(dir);
		traverse();

	}

	public static void updateJavadoc(String classname) throws Exception {

		AllJavadoc jdoc = new AllJavadoc();
		jdoc.setClassname(classname);
		jdoc.setDir(dir);
		jdoc.setUseStars(false);

		String result = "";

		try {
			result = jdoc.updateJavadoc();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		//System.out.println(result + "\n");
		
		File file;
		BufferedWriter writer;

		if (!result.isEmpty()) {
			file = new File(jdoc.getDir() + "/"
					+ jdoc.getClassname().replaceAll("\\.", "/") + ".java");
			if (!file.exists()) {
				System.out.println("File '" + file.getAbsolutePath()
						+ "' doesn't exist!");
				return;
			}

			writer = new BufferedWriter(new FileWriter(file));
			writer.write(result);
			writer.close();

			System.out.println(jdoc.getClassname()
					+ "'s Javadoc successfully updated.");

			return;
		} else {
			System.out.println(jdoc.getClassname()
					+ "'s Javadoc update failed. Skipping file..");
		}
		
	}

	public static void traverse() throws Exception {
		recursiveTraversal(originalDir);
	}

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

	public String globalInfo() {
		return "This class uses weka's Javadoc auto-generation classes to generate Javadoc\n"
				+ "comments and replaces the content between certain comment tags.";
	}

	@Override
	public String[] getOptions() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	/*
	 * * Returns an enumeration describing the available options.
	 * 
	 * @return an enumeration of all the available options.
	 */
	public Enumeration listOptions() {

		Vector result = new Vector(1);

		result.addElement(new Option(
				"\tThe directory where the mulan package resides.",
				"dir", 1, "-dir <dir>"));

		return result.elements();
	}

	@Override
	public void setOptions(String[] options) throws Exception {
		// TODO Auto-generated method stub

	}
}