package WEKA;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Date;
import java.util.Random;

public class PRAKTIKA4_ARIKETA1 {

    private static Instances data;
    private static PrintWriter printWriter;

    public static void main(String[] args) throws Exception {
        if(args.length != 3) {
            System.out.println("EZ DITUZU BEHAR DIREN PARAMETROAK SARTU");
            System.out.println("HURRENGO MODUAN JARRI BEHARKO DUZU:");
            System.out.println("java -jar EreduaSortu.jar  /paht/to/data.arff  /paht/to/irteerako/karpeta/NB.model  /paht/to/irteerako/karpeta/KalitatearenEstimazioa.txt");
        }else{
            FileWriter fileWriter = new FileWriter(args[2]);
            printWriter = new PrintWriter(fileWriter);
            printWriter.println("EGIKARITUTAKO DATA -> " + new Date());
            printWriter.println("LEHENENGO ARGUMENTUA: ARFF FITXATEGIA -> " + args[0]);
            printWriter.println("BIGARREN ARGUMENTUA: NAIVEBAYES MODELOA GORDETZEKO HELBIDEA -> " + args[1]);
            printWriter.println("HIRUGARREN ARGUMENTUA: ESTIMAZIOKO DATUAK -> " + args[2] + "\n");
            datuakKargatu(args[0]);
            ariketaEgin(args[1]);
            printWriter.close();
            fileWriter.close();
        }
    }

    private static void ariketaEgin(String path) throws Exception {
        NaiveBayes naiveBayes = new NaiveBayes();
        naiveBayes.buildClassifier(data);
        //OBJETUA GORDE
        SerializationHelper.write(path,naiveBayes);
        kalitateaEstimatu(naiveBayes);
    }

    private static void kalitateaEstimatu(Classifier naive) throws Exception {
        //CROSS VALIDATION
        Evaluation crossValidation = new Evaluation(data);
        crossValidation.crossValidateModel(naive,data,10,new Random(1));
        printWriter.println("CROSS VALIDATION --> ");
        printWriter.println("ACCURACY -> " + crossValidation.pctCorrect());
        printWriter.println("CONFUSION MATRIX -> " + crossValidation.toMatrixString() + "\n");

        //HOLD-OUT
        Randomize randomize = new Randomize();
        randomize.setRandomSeed(1);
        randomize.setInputFormat(data);
        Instances random = Filter.useFilter(data,randomize);

        RemovePercentage removePercentageTrain = new RemovePercentage();
        removePercentageTrain.setInputFormat(random);
        removePercentageTrain.setInvertSelection(true);
        removePercentageTrain.setPercentage(70.0);
        Instances train = Filter.useFilter(random,removePercentageTrain);

        RemovePercentage removePercentageTest = new RemovePercentage();
        removePercentageTest.setInputFormat(random);
        removePercentageTest.setInvertSelection(false);
        removePercentageTest.setPercentage(70.0);
        Instances test = Filter.useFilter(random,removePercentageTest);

        Evaluation holdOut = new Evaluation(train);
        double[] predictions = holdOut.evaluateModel(naive,test);
        printWriter.println("HOLD-OUT --> ");
        printWriter.println("ACCURACY -> " + holdOut.pctCorrect());
        printWriter.println("CONFUSION MATRIX -> " + holdOut.toMatrixString());
    }

    private static void datuakKargatu(String arffFtixategia) throws Exception {
        DataSource source = new DataSource(arffFtixategia);
        data = source.getDataSet();
        if(data.classIndex() == -1) data.setClassIndex(data.numAttributes()-1);
    }
}
