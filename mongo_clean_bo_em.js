conn = new Mongo();
db = conn.getDB("spearmint");
db['bo-em-hmm.jobs'].drop();
db['bo-em-hmm.hypers'].drop();