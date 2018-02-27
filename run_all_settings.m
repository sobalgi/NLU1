function run_all_settings
%%  NLU Assignment 1
%   Assignment 1  - Natural Language model
%
%   Author           : Sourabh V Balgi.
%   Sr. No.          : 04-03-02-10-42-17-1-14318.
%   Dept             : EE.
%   Course           : M. Tech. (Systems Engineering).
%   Created on       : 15-Feb-2018.

% House Keeping
clc
close all
global D1_trn_ix_b D1_dev_ix_b D1_tst_ix_b D1_trn_u D1_trn_b D1_dev_b D1_tst_b D1_pu D1_pb D1_trn_loc D1_dev_loc D1_tst_loc D1_vocab Num_D1_vocab %#ok<NUSED>
global D2_trn_ix_b D2_dev_ix_b D2_tst_ix_b D2_trn_u D2_trn_b D2_dev_b D2_tst_b D2_pu D2_pb D2_trn_loc D2_dev_loc D2_tst_loc D2_vocab Num_D2_vocab %#ok<NUSED>
global  D1_trn D1_dev D1_tst D2_trn D2_dev D2_tst %#ok<NUSED>

%% S1: Train: D1-Train, Test: D1-Test
disp('>>> running S1: Train: D1-Train, Test: D1-Test');
setting_s1
%% S2: Train: D2-Train, Test: D2-Test
disp('>>> running S2: Train: D2-Train, Test: D2-Test');
setting_s2
%% S3: Train: D1-Train + D2-Train, Test: D1-Test
disp('>>> running S3: Train: D1-Train + D2-Train, Test: D1-Test');
setting_s3
%% S4: Train: D1-Train + D2-Train, Test: D2-Test
disp('>>> running S4: Train: D1-Train + D2-Train, Test: D2-Test');
setting_s4

end

%% S1: Train: D1-Train, Test: D1-Test
function setting_s1
global D1_trn_ix_b D1_dev_ix_b D1_tst_ix_b D1_trn_u D1_trn_b D1_dev_b D1_tst_b D1_pu D1_pb D1_trn_loc D1_dev_loc D1_tst_loc D1_vocab Num_D1_vocab
global  D1_trn D1_dev D1_tst

tic
%% Compute the n-gram counts fron the Datafiles
if isempty(D1_trn_u)||isempty(D1_trn_b)%||isempty(D1_trn_t)% clear
    
    cur_dir = pwd;
    D1_pth = [cur_dir '/brown'];
    addpath(D1_pth);
    
    %% Train, Dev and Test Split and Preprocessing (Brown Corpus)
    D1_cat = readtable([D1_pth '/cats.txt'],'ReadVariableNames',false);
    D1_cat.Properties.VariableNames = {'filename','category'};
    D1_cat.category = categorical(D1_cat.category);
    D1_fn = ls(D1_pth);
    %     D1_trn = '<\s\> ';% 85% Train Split
    %     D1_dev = '<\s\> ';% 5% Dev   Split
    %     D1_tst = '<\s\> ';% 10% Test  Split
    D1_trn = '';% 85% Train Split
    D1_dev = '';% 5% Dev   Split
    D1_tst = '';% 10% Test  Split
    for i=97:114 % char(97)- a , char(114) - r
        fn_i = sort(regexp(D1_fn,['c' char(i) '\d+'],'match'));
        num_fn_i = length(fn_i);
        for j=1:num_fn_i %
            this_str = fileread([D1_pth '/' fn_i{j}]);
            this_str = lower(this_str);
            sents = regexp(this_str,' \./\.\s*');
            n_sents = length(sents);
            trn_end_idx = sents(floor(n_sents*.70))+3;
            dev_end_idx = sents(floor(n_sents*.80))+3;
            this_str_trn = this_str(1:trn_end_idx);
            this_str_dev = this_str(trn_end_idx:dev_end_idx);
            this_str_tst = this_str(dev_end_idx:end);
            this_str_trn = regexprep(this_str_trn,['/\w*[' char([33:38]) '\' char(39) char([40:45]) char([46:47 58:64 91]) '\\' char([93:96 123:126]) ']*\w* '],' ');
            %this_str_trn = regexprep(this_str_trn,'\./\.','. <\\s\\> ');% sentence break tag
            this_str_trn = regexprep(this_str_trn,'\s+',' ');
            this_str_trn = regexprep(this_str_trn,'-hl','');
            %             this_str_trn = regexprep(this_str_trn,'^\s+','');
            this_str_trn = regexprep(this_str_trn,'!/\.','!');
            this_str_trn = regexprep(this_str_trn,['[!"#$%&\''\(\)\*\+,-/:;<=>\?\@\[\\\]\^_\`\{\|\}~]'],'');
            D1_trn = [D1_trn this_str_trn];
            %             D1_trn_doc{end+1,1} = this_str;
            %             D1_cat.split(count) = categorical({'trn'});
            this_str_dev = regexprep(this_str_dev,['/\w*[' char([33:38]) '\' char(39) char([40:45]) '\' char([46:47 58:64 91]) '\\' char([93:96 123:126]) ']*\w* '],' ');
            %this_str_dev = regexprep(this_str_dev,'\./\.','. <\\s\\> ');% sentence break tag
            this_str_dev = regexprep(this_str_dev,'\s+',' ');
            this_str_dev = regexprep(this_str_dev,'-hl','');
            %             this_str_dev = regexprep(this_str_dev,'^\s+','');
            this_str_dev = regexprep(this_str_dev,'!/\.','!');
            this_str_dev = regexprep(this_str_dev,['[!"#$%&\''\(\)\*\+,-/:;<=>\?\@\[\\\]\^_\`\{\|\}~]'],'');
            D1_dev = [D1_dev this_str_dev];
            %             D1_dev_doc{end+1,1} = this_str;
            %             D1_cat.split(count) = categorical({'dev'});
            this_str_tst = regexprep(this_str_tst,['/\w*[' char([33:38]) '\' char(39) char([40:45]) char([46:47 58:64 91]) '\\' char([93:96 123:126]) ']*\w* '],' ');
            %this_str_tst = regexprep(this_str_tst,'\./\.','. <\\s\\> ');% sentence break tag
            this_str_tst = regexprep(this_str_tst,'\s+',' ');
            this_str_tst = regexprep(this_str_tst,'-hl','');
            %             this_str_tst = regexprep(this_str_tst,'^\s+','');
            this_str_tst = regexprep(this_str_tst,'!/\.','!');
            this_str_tst = regexprep(this_str_tst,['[!"#$%&\''\(\)\*\+,-/:;<=>\?\@\[\\\]\^_\`\{\|\}~]'],'');
            D1_tst = [D1_tst this_str_tst];
            %             D1_tst_doc{end+1,1} = this_str;
            %             D1_cat.split(count) = categorical({'tst'});
            %             count = count+1;
        end
    end
    
    %% Unigram - trn
    D1_trn_u_tkn = regexp(D1_trn,'\S+','match');%( )?
    D1_vocab = [unique(D1_trn_u_tkn,'stable') {'<UNK>'}];% adding extra Unknown word to Vocab
    Num_D1_vocab = length(D1_vocab);
    [~,D1_trn_loc] = ismember(D1_trn_u_tkn,D1_vocab);
    D1_trn_loc_str = regexprep(num2str(D1_trn_loc),'\s+',' ');
    space_idx = regexp(D1_trn_loc_str,' ');
    D1_trn_u = hist(D1_trn_loc,1:Num_D1_vocab)';
    
    %% Bigram - trn
    D1_2a_tkn = regexp(D1_trn_loc_str,'\d+ \d+','match');
    D1_2b_tkn = regexp(D1_trn_loc_str(space_idx(1)+1:end),'\d+ \d+','match');% as overlapping find is not
    D1_trn_b_tkn = [D1_2a_tkn D1_2b_tkn];
    vocab2 = unique(D1_trn_b_tkn,'stable');
    Num_vocab2 = length(vocab2);
    [~,vocab2_loc] = ismember(D1_trn_b_tkn,vocab2);
    D1_trn_b = hist(vocab2_loc,1:Num_vocab2)';
    D1_trn_ix_b = cell2mat(cellfun(@str2num,vocab2','UniformOutput',false));
    
    %     %% Trigram - trn
    %     D1_3a_tkn = regexp(D1_trn_loc_str,'\d+ \d+ \d+','match');
    %     D1_3b_tkn = regexp(D1_trn_loc_str(space_idx(1)+1:end),'\d+ \d+ \d+','match');% as overlapping find is not
    %     D1_3c_tkn = regexp(D1_trn_loc_str(space_idx(2)+1:end),'\d+ \d+ \d+','match');% as overlapping find is not
    %     D1_trn_t_tkn = [D1_3a_tkn D1_3b_tkn D1_3c_tkn];
    %     vocab3 = unique(D1_trn_t_tkn,'stable');
    %     Num_vocab3 = length(vocab3);
    %     [~,vocab3_loc] = ismember(D1_trn_t_tkn,vocab3);
    %     D1_trn_t = hist(vocab3_loc,1:Num_vocab3)';
    %     D1_trn_ix_t = cell2mat(cellfun(@str2num,vocab3','UniformOutput',false));
    
    %% Unigram - dev
    D1_dev_u_tkn = regexp(D1_dev,'\S+','match');%( )?
    [~,D1_dev_loc] = ismember(D1_dev_u_tkn,D1_vocab);
    D1_dev_loc(D1_dev_loc==0)=Num_D1_vocab;% Unknown
    D1_dev_loc_str = regexprep(num2str(D1_dev_loc),'\s+',' ');
    space_idx = regexp(D1_dev_loc_str,' ');
    
    %% Bigram - dev
    D1_2a_tkn = regexp(D1_dev_loc_str,'\d+ \d+','match');
    D1_2b_tkn = regexp(D1_dev_loc_str(space_idx(1)+1:end),'\d+ \d+','match');% as overlapping find is not
    D1_dev_2_tkn = [D1_2a_tkn D1_2b_tkn];
    [~,vocab2_loc] = ismember(D1_dev_2_tkn,vocab2);
    D1_dev_b = hist(vocab2_loc,1:Num_vocab2)';
    D1_dev_ix_b = cell2mat(cellfun(@str2num,D1_dev_2_tkn','UniformOutput',false));
    
    %     %% Trigram - dev
    %     D1_3a_tkn = regexp(D1_dev_loc_str,'\d+ \d+ \d+','match');
    %     D1_3b_tkn = regexp(D1_dev_loc_str(space_idx(1)+1:end),'\d+ \d+ \d+','match');% as overlapping find is not
    %     D1_3c_tkn = regexp(D1_dev_loc_str(space_idx(2)+1:end),'\d+ \d+ \d+','match');% as overlapping find is not
    %     D1_dev_t_tkn = [D1_3a_tkn D1_3b_tkn D1_3c_tkn];
    %     [~,vocab3_loc] = ismember(D1_dev_t_tkn,vocab3);
    %     D1_dev_t = hist(vocab3_loc,1:Num_vocab3)';
    %     D1_dev_ix_t = cell2mat(cellfun(@str2num,D1_dev_t_tkn','UniformOutput',false));
    
    %% Unigram - tst
    D1_tst_u_tkn = regexp(D1_tst,'\S+','match');%( )?
    [~,D1_tst_loc] = ismember(D1_tst_u_tkn,D1_vocab);
    D1_tst_loc(D1_tst_loc==0)=Num_D1_vocab;% Unknown
    D1_tst_loc_str = regexprep(num2str(D1_tst_loc),'\s+',' ');
    space_idx = regexp(D1_tst_loc_str,' ');
    
    %% Bigram -tst
    D1_2a_tkn = regexp(D1_tst_loc_str,'\d+ \d+','match');
    D1_2b_tkn = regexp(D1_tst_loc_str(space_idx(1)+1:end),'\d+ \d+','match');% as overlapping find is not
    D1_tst_b_tkn = [D1_2a_tkn D1_2b_tkn];
    [~,vocab2_loc] = ismember(D1_tst_b_tkn,vocab2);
    D1_tst_b = hist(vocab2_loc,1:Num_vocab2)';
    D1_tst_ix_b = cell2mat(cellfun(@str2num,D1_tst_b_tkn','UniformOutput',false));
    
    %     %% Trigram - tst
    %     D1_3a_tkn = regexp(D1_tst_loc_str,'\d+ \d+ \d+','match');
    %     D1_3b_tkn = regexp(D1_tst_loc_str(space_idx(1)+1:end),'\d+ \d+ \d+','match');% as overlapping find is not
    %     D1_3c_tkn = regexp(D1_tst_loc_str(space_idx(2)+1:end),'\d+ \d+ \d+','match');% as overlapping find is not
    %     D1_tst_t_tkn = [D1_3a_tkn D1_3b_tkn D1_3c_tkn];
    %     [~,vocab3_loc] = ismember(D1_tst_t_tkn,vocab3);
    %     D1_tst_t = hist(vocab3_loc,1:Num_vocab3)';
    %     D1_tst_ix_t = cell2mat(cellfun(@str2num,D1_tst_t_tkn','UniformOutput',false));
    
end
toc

%% Laplace Smoothing
% new_C1 = C1+1;
% D1_pw1 = log(new_C1/sum(new_C1));
% Observation :
% since vocabulary is very large, Add one smoothing (Laplace smoothing)
% does not help here as adding 1 to numerator has negligible effect when
% dividing by Num_D1_vocab = 44674. Still the prob is very low. Infact we have
% even made it worse for probabilities good counts earlier.
% Even Add k smoothing also produces same result because of same large
% Num_Vocab = 44674 words
% So We check the criteria for using Good-Turing Smoothing.

%% Good-Turing Smoothing
% [max_C1,max_C1_idx] = max(D1_trn_C1);
% % C1_GT=C1;
% % C1_GT(C1>=3)=3;
% % max_C1 = 3;
% Num_C1 = hist(D1_trn_C1,1:max_C1);
% zero_Num_C1_idx = find(Num_C1==0);
% % Observation
% % since max_C1 = 42603(the), is very large, there are lots of C
% % which have zero Num_C "zero_Num_C1_idx". So Good-Turing Smoothing cannot be used here.
% % So we move to Kneser-Ney smoothing

%% We use modified Kneser-Ney
%% Unigram model N=1
% optimization for d in unigram using held out set
Num_D1_vocab = length(D1_vocab);
% d=.1:.1:.9;
% for i=1:length(d)
% d1(1) = d(i);
% d1(2) = d(i);
% d1(3) = d(i);
%
% D1_pu = max(D1_trn_u - (D1_trn_u==1)*d1(1)- (D1_trn_u==2)*d1(2)- (D1_trn_u>=3)*d1(3),0)/sum(D1_trn_u);
% % D1_pu = D1_pu+(abs(1-sum(D1_pu))/Num_D1_vocab);
% D1_pu(end) = 1-sum(D1_pu);% <unk> prob
%
% % find dev perplexity
% D1_dev_ppl_u = exp(-1/length(D1_dev_loc)*sum(log(D1_pu(D1_dev_loc))));
% disp(['optim : S1 Dev Data Unigram Perplexity is ' num2str(D1_dev_ppl_u) ' for d = ' num2str(d(i))]);
%
% end

%% Post optimization unigram model
d=.5;
d1(1) = d;
d1(2) = d;
d1(3) = d;

D1_pu = max(D1_trn_u - (D1_trn_u==1)*d1(1)- (D1_trn_u==2)*d1(2)- (D1_trn_u>=3)*d1(3),0)/sum(D1_trn_u);
% D1_pu = D1_pu+(abs(1-sum(D1_pu))/Num_D1_vocab);
D1_pu(end) = 1-sum(D1_pu);% <unk> prob

% find dev perplexity
D1_dev_ppl_u = exp(-1/length(D1_dev_loc)*sum(log(D1_pu(D1_dev_loc))));
disp(['Result : S1 Dev Data Unigram Perplexity is ' num2str(D1_dev_ppl_u) ' for d = ' num2str(d)]);


% test perplexity Unigram
D1_tst_ppl_u = exp(-1/length(D1_tst_loc)*sum(log(D1_pu(D1_tst_loc))));
disp(['Result : S1 Test Data Unigram Perplexity is ' num2str(D1_tst_ppl_u) ' for d = ' num2str(d)]);
disp(newline);

%% Bigram model N=2
%% modified unigram for bigram
a_w1w2 = sparse(D1_trn_ix_b(:,1),D1_trn_ix_b(:,2),D1_trn_b,Num_D1_vocab,Num_D1_vocab);
a_xw2 = sum(a_w1w2>0);
D1_trn_u_mod = full(a_xw2)';

% d=.1:.1:.9;
% for i=1:length(d)
% d1(1) = d(i);
% d1(2) = d(i);
% d1(3) = d(i);
% % d=.2;
% % d1(1) = d;
% % d1(2) = d;
% % d1(3) = d;
%
% D1_pu = max(D1_trn_u_mod - (D1_trn_u_mod==1)*d1(1)- (D1_trn_u_mod==2)*d1(2)- (D1_trn_u_mod>=3)*d1(3),0)/sum(D1_trn_u_mod);
% % D1_pu = D1_pu+(abs(1-sum(D1_pu))/Num_D1_vocab);
% D1_pu(end) = 1-sum(D1_pu);% <unk> prob
%
% D1_dev_ppl_u = exp(-1/length(D1_dev_loc)*sum(log(D1_pu(D1_dev_loc))));
% disp(['optim : S1 Dev Data Unigram Perplexity is ' num2str(D1_dev_ppl_u) ' for d = ' num2str(d(i))]);
%
% end

d=.4;
d1(1) = d;
d1(2) = d;
d1(3) = d;

D1_pu = max(D1_trn_u_mod - (D1_trn_u_mod==1)*d1(1)- (D1_trn_u_mod==2)*d1(2)- (D1_trn_u_mod>=3)*d1(3),0)/sum(D1_trn_u_mod);
% D1_pu = D1_pu+(abs(1-sum(D1_pu))/Num_D1_vocab);
D1_pu(end) = 1-sum(D1_pu);% <unk> prob

%% Post Optimization Bigram model
% d=.9:.01:.99;
% for i=1:length(d)
% d2(1) = d(i);
% d2(2) = d(i);
% d2(3) = d(i);
% % d=.8;
% % d2(1) = d;
% % d2(2) = d;
% % d2(3) = d;
%
% D1_pb = max(D1_trn_b - (D1_trn_b==1)*d2(1)- (D1_trn_b==2)*d2(2)- (D1_trn_b>=3)*d2(3),0)./D1_trn_u(D1_trn_ix_b(:,1));
% a_w1x = sparse(D1_trn_ix_b(:,1),D1_trn_ix_b(:,2),D1_trn_b,Num_D1_vocab,Num_D1_vocab);
% b_w1 = full(sum(a_w1x==1,2)*d2(1)+ sum(a_w1x==2,2)*d2(2)+ sum(a_w1x==3,2)*d2(3));%./D1_trn_u(D1_trn_ix_b(:,2))
% D1_pb = D1_pb + b_w1(D1_trn_ix_b(:,1)).*D1_pu(D1_trn_ix_b(:,2))./D1_trn_u(D1_trn_ix_b(:,1));
% D1_pb_mat = sparse(D1_trn_ix_b(:,1),D1_trn_ix_b(:,2),D1_pb,Num_D1_vocab,Num_D1_vocab);
% D1_pb_norm = (1-sum(D1_pb_mat,2));
%
% [D1_pb_exists,D1_pb_exists_ix] = ismember(D1_dev_ix_b,D1_trn_ix_b,'rows');
% D1_pb_dev_ix = D1_pb_exists_ix(D1_pb_exists);
% sum_log_p = sum(log(D1_pb(D1_pb_dev_ix)))+sum(log(D1_pb_norm(D1_dev_ix_b(~D1_pb_exists,1)).*D1_pu(D1_dev_ix_b(~D1_pb_exists,2))));
% D1_dev_ppl_b = exp(-sum_log_p/length(D1_dev_ix_b));
% disp(['optim : S1 Dev Data Bigram Perplexity is ' num2str(D1_dev_ppl_b) ' for d = ' num2str(d(i))]);
%
% end

d=.9;
d2(1) = d;
d2(2) = d;
d2(3) = d;

D1_pb = max(D1_trn_b - (D1_trn_b==1)*d2(1)- (D1_trn_b==2)*d2(2)- (D1_trn_b>=3)*d2(3),0)./D1_trn_u(D1_trn_ix_b(:,1));
a_w1x = sparse(D1_trn_ix_b(:,1),D1_trn_ix_b(:,2),D1_trn_b,Num_D1_vocab,Num_D1_vocab);
b_w1 = full(sum(a_w1x==1,2)*d2(1)+ sum(a_w1x==2,2)*d2(2)+ sum(a_w1x==3,2)*d2(3));%./D1_trn_u(D1_trn_ix_b(:,2))
D1_pb = D1_pb + b_w1(D1_trn_ix_b(:,1)).*D1_pu(D1_trn_ix_b(:,2))./D1_trn_u(D1_trn_ix_b(:,1));
D1_pb_mat = sparse(D1_trn_ix_b(:,1),D1_trn_ix_b(:,2),D1_pb,Num_D1_vocab,Num_D1_vocab);
D1_pb_norm = (1-sum(D1_pb_mat,2));

[D1_pb_exists,D1_pb_exists_ix] = ismember(D1_dev_ix_b,D1_trn_ix_b,'rows');
D1_pb_dev_ix = D1_pb_exists_ix(D1_pb_exists);
sum_log_p = sum(log(D1_pb(D1_pb_dev_ix)))+sum(log(D1_pb_norm(D1_dev_ix_b(~D1_pb_exists,1)).*D1_pu(D1_dev_ix_b(~D1_pb_exists,2))));
D1_dev_ppl_b = exp(-sum_log_p/length(D1_dev_ix_b));
disp(['Result : S1 Dev Data Bigram Perplexity is ' num2str(D1_dev_ppl_b) ' for d = ' num2str(d)]);

[D1_pb_exists,D1_pb_exists_ix] = ismember(D1_tst_ix_b,D1_trn_ix_b,'rows');
D1_pb_tst_ix = D1_pb_exists_ix(D1_pb_exists);
sum_log_p = sum(log(D1_pb(D1_pb_tst_ix)))+sum(log(D1_pb_norm(D1_tst_ix_b(~D1_pb_exists,1)).*D1_pu(D1_tst_ix_b(~D1_pb_exists,2))));
D1_tst_ppl_b = exp(-sum_log_p/length(D1_tst_ix_b));
disp(['Result : S1 Test Data Bigram Perplexity is ' num2str(D1_tst_ppl_b) ' for d = ' num2str(d)]);
disp(newline);

%% Bigram Sentence generation
w1=[];
str ='';%<\s\>
for i=1:9
    num_toD1_pwords = randi(10);
    if isempty(w1)
        % [~,w1_ix] = mink(D1_pu(1:end-1),10000);
        % w1 = w1_ix(randi(length(w1_ix)));
        w1 = randi(Num_D1_vocab-1);
        % w1 = w1_ix(end);
        str = D1_vocab{w1};
    end
    w1_ix = find(D1_trn_ix_b(:,1)==w1);
    [max_D1_pb,w2_ix] = maxk(D1_pb(w1_ix),num_toD1_pwords);% D1_trn_ix_b(w1_ix,2)
    rand_toD1_pk = randi(length(w2_ix));
    max_D1_pb = max_D1_pb(rand_toD1_pk);
    [max_D1_pb_n,norm_ix] = maxk(D1_pb_norm(w1)*D1_pu(1:end-1),num_toD1_pwords);% D1_trn_ix_b(w1_ix,2)
    max_D1_pb_n = max_D1_pb_n(randi(length(norm_ix)));
    if max_D1_pb > max_D1_pb_n
        w2 = D1_trn_ix_b(w1_ix(w2_ix(rand_toD1_pk)),2);
    else
        w2 = norm_ix(randi(length(norm_ix)));
    end
    str = [str ' ' D1_vocab{w2}];
    w1 = w2;
end
disp(['Sentence generated from S1 Setting - "' str ' "'])
disp(newline);

end

%% S2: Train: D2-Train, Test: D2-Test
function setting_s2
%%  NLU Assignment 1
%   Assignment 1  - Natural Language model
%
%   Author           : Sourabh V Balgi.
%   Sr. No.          : 04-03-02-10-42-17-1-14318.
%   Dept             : EE.
%   Course           : M. Tech. (Systems Engineering).
%   Created on       : 15-Feb-2018.

% House Keeping
global D2_trn_ix_b D2_dev_ix_b D2_tst_ix_b D2_trn_u D2_trn_b D2_dev_b D2_tst_b D2_pu D2_pb D2_trn_loc D2_dev_loc D2_tst_loc D2_vocab Num_D2_vocab
global D2_trn D2_dev D2_tst

tic
%% Compute the n-gram counts fron the Datafiles
if isempty(D2_trn_u)||isempty(D2_trn_b)% clear
    
    cur_dir = pwd;
    D2_pth = [cur_dir '/gutenberg'];
    addpath(D2_pth);
    
    %% Train, Dev and Test Split and Preprocessing (Gutenberg Corpus)
    D2_fn = ls(D2_pth);
    %     D2_trn = '<\s\> ';% 85% Train Split
    %     D2_dev = '<\s\> ';% 5% Dev   Split
    %     D2_tst = '<\s\> ';% 10% Test  Split
    D2_trn = '';% 85% Train Split
    D2_dev = '';% 5% Dev   Split
    D2_tst = '';% 10% Test  Split
    %%
    fn_i = sort(regexp(D2_fn,['\w+-\w+.txt'],'match'));
    num_fn_i = length(fn_i);
    for j=1:num_fn_i %
        this_str = fileread([D2_pth '/' fn_i{j}]);
        this_str = lower(this_str);
        this_str = regexprep(this_str,'\d*:\d*','');%bible
        sents = regexp(this_str,'(\.|")\s*');
        n_sents = length(sents);
        trn_end_idx = sents(floor(n_sents*.70))+1;
        dev_end_idx = sents(floor(n_sents*.80))+1;
        this_str = regexprep(this_str,'\s+',' ');%bible
        this_str_trn = this_str(1:trn_end_idx);
        this_str_dev = this_str(trn_end_idx:dev_end_idx);
        this_str_tst = this_str(dev_end_idx:end);% '[!"#$%&\''\(\)\*\+,-/:;<=>\?\@\[\\\]\^_\`\{\|\}~]'
        this_str_trn = regexprep(this_str_trn,['[!"#$%&\''\(\)\*\+,-/:;<=>\?\@\[\\\]\^_\`\{\|\}~]'],'');
        % this_str_trn = regexprep(this_str_trn,['[' char([33:38]) '\' char(39) char([40:45]) char([47 58:64 91]) '\\' char([93:96 123:126]) ']'],'');
        this_str_trn = regexprep(this_str_trn,'\.\s*','. <\s\> ');
        D2_trn = [D2_trn this_str_trn];
        this_str_dev = regexprep(this_str_dev,['[!"#$%&\''\(\)\*\+,-/:;<=>\?\@\[\\\]\^_\`\{\|\}~]'],'');
        %        this_str_dev = regexprep(this_str_dev,['[' char([33:38]) '\' char(39) char([40:45]) char([47 58:64 91]) '\\' char([93:96 123:126]) ']'],'');
        this_str_dev = regexprep(this_str_dev,'\.\s*','. <\s\> ');% sentence break tag
        D2_dev = [D2_dev this_str_dev];
        this_str_tst = regexprep(this_str_tst,['[!"#$%&\''\(\)\*\+,-/:;<=>\?\@\[\\\]\^_\`\{\|\}~]'],'');
        %        this_str_tst = regexprep(this_str_tst,['[' char([33:38]) '\' char(39) char([40:45]) char([47 58:64 91]) '\\' char([93:96 123:126]) ']'],'');
        this_str_tst = regexprep(this_str_tst,'\.\s*','. <\s\> ');% sentence break tag
        D2_tst = [D2_tst this_str_tst];
    end
    
    %% Unigram - trn
    D2_trn_u_tkn = regexp(D2_trn,'\S+','match');%( )?
    D2_vocab = [unique(D2_trn_u_tkn,'stable') {'<UNK>'}];% adding extra Unknown word to Vocab
    Num_D2_vocab = length(D2_vocab);
    [~,D2_trn_loc] = ismember(D2_trn_u_tkn,D2_vocab);
    D2_trn_loc_str = regexprep(num2str(D2_trn_loc),'\s+',' ');
    space_idx = regexp(D2_trn_loc_str,' ');
    D2_trn_u = hist(D2_trn_loc,1:Num_D2_vocab)';
    
    %% Bigram - trn
    D2_2a_tkn = regexp(D2_trn_loc_str,'\d+ \d+','match');
    D2_2b_tkn = regexp(D2_trn_loc_str(space_idx(1)+1:end),'\d+ \d+','match');% as overlapping find is not
    D2_trn_b_tkn = [D2_2a_tkn D2_2b_tkn];
    vocab2 = unique(D2_trn_b_tkn,'stable');
    Num_vocab2 = length(vocab2);
    [~,vocab2_loc] = ismember(D2_trn_b_tkn,vocab2);
    D2_trn_b = hist(vocab2_loc,1:Num_vocab2)';
    D2_trn_ix_b = cell2mat(cellfun(@str2num,vocab2','UniformOutput',false));
    
    %% Unigram - dev
    D2_dev_u_tkn = regexp(D2_dev,'\S+','match');%( )?
    [~,D2_dev_loc] = ismember(D2_dev_u_tkn,D2_vocab);
    D2_dev_loc(D2_dev_loc==0)=Num_D2_vocab;% Unknown
    D2_dev_loc_str = regexprep(num2str(D2_dev_loc),'\s+',' ');
    space_idx = regexp(D2_dev_loc_str,' ');
    
    %% Bigram - dev
    D2_2a_tkn = regexp(D2_dev_loc_str,'\d+ \d+','match');
    D2_2b_tkn = regexp(D2_dev_loc_str(space_idx(1)+1:end),'\d+ \d+','match');% as overlapping find is not
    D2_dev_2_tkn = [D2_2a_tkn D2_2b_tkn];
    [~,vocab2_loc] = ismember(D2_dev_2_tkn,vocab2);
    D2_dev_b = hist(vocab2_loc,1:Num_vocab2)';
    D2_dev_ix_b = cell2mat(cellfun(@str2num,D2_dev_2_tkn','UniformOutput',false));
    
    %% Unigram - tst
    D2_tst_u_tkn = regexp(D2_tst,'\S+','match');%( )?
    [~,D2_tst_loc] = ismember(D2_tst_u_tkn,D2_vocab);
    D2_tst_loc(D2_tst_loc==0)=Num_D2_vocab;% Unknown
    D2_tst_loc_str = regexprep(num2str(D2_tst_loc),'\s+',' ');
    space_idx = regexp(D2_tst_loc_str,' ');
    
    %% Bigram -tst
    D2_2a_tkn = regexp(D2_tst_loc_str,'\d+ \d+','match');
    D2_2b_tkn = regexp(D2_tst_loc_str(space_idx(1)+1:end),'\d+ \d+','match');% as overlapping find is not
    D2_tst_b_tkn = [D2_2a_tkn D2_2b_tkn];
    [~,vocab2_loc] = ismember(D2_tst_b_tkn,vocab2);
    D2_tst_b = hist(vocab2_loc,1:Num_vocab2)';
    D2_tst_ix_b = cell2mat(cellfun(@str2num,D2_tst_b_tkn','UniformOutput',false));
    
end
toc

%% Laplace Smoothing
% new_C1 = C1+1;
% D2_pw1 = log(new_C1/sum(new_C1));
% Observation :
% since vocabulary is very large, Add one smoothing (Laplace smoothing)
% does not help here as adding 1 to numerator has negligible effect when
% dividing by Num_D2_vocab = 44674. Still the prob is very low. Infact we have
% even made it worse for probabilities good counts earlier.
% Even Add k smoothing also produces same result because of same large
% Num_Vocab = 44674 words
% So We check the criteria for using Good-Turing Smoothing.

%% Good-Turing Smoothing
% [max_C1,max_C1_idx] = max(D2_trn_C1);
% % C1_GT=C1;
% % C1_GT(C1>=3)=3;
% % max_C1 = 3;
% Num_C1 = hist(D2_trn_C1,1:max_C1);
% zero_Num_C1_idx = find(Num_C1==0);
% % Observation
% % since max_C1 = 42603(the), is very large, there are lots of C
% % which have zero Num_C "zero_Num_C1_idx". So Good-Turing Smoothing cannot be used here.
% % So we move to Kneser-Ney smoothing

%% We use modified Kneser-Ney
%% Unigram model N=1
% optimization for d in unigram using held out set
Num_D2_vocab = length(D2_vocab);
%% Post optimization unigram model
% d=.5;
% D2(1) = d;
% D2(2) = d;
% D2(3) = d;

% d=.1:.1:.9;
% for i=1:length(d)
% D2(1) = d(i);
% D2(2) = d(i);
% D2(3) = d(i);
%
% D2_pu = max(D2_trn_u - (D2_trn_u==1)*D2(1)- (D2_trn_u==2)*D2(2)- (D2_trn_u>=3)*D2(3),0)/sum(D2_trn_u);
% % D2_pu = D2_pu+(abs(1-sum(D2_pu))/Num_D2_vocab);
% D2_pu(end) = 1-sum(D2_pu);% <unk> prob
%
% % find dev perplexity
% D2_dev_ppl_u = exp(-1/length(D2_dev_loc)*sum(log(D2_pu(D2_dev_loc))));
% disp(['optim : S2 Dev Data Unigram Perplexity is ' num2str(D2_dev_ppl_u) ' for d = ' num2str(d(i))]);
%
% end

d=.5;
D2(1) = d;
D2(2) = d;
D2(3) = d;

D2_pu = max(D2_trn_u - (D2_trn_u==1)*D2(1)- (D2_trn_u==2)*D2(2)- (D2_trn_u>=3)*D2(3),0)/sum(D2_trn_u);
% D2_pu = D2_pu+(abs(1-sum(D2_pu))/Num_D2_vocab);
D2_pu(end) = 1-sum(D2_pu);% <unk> prob

% find dev perplexity
D2_dev_ppl_u = exp(-1/length(D2_dev_loc)*sum(log(D2_pu(D2_dev_loc))));
disp(['Result : S2 Dev Data Unigram Perplexity is ' num2str(D2_dev_ppl_u) ' for d = ' num2str(d)]);

% test perplexity Unigram
D2_tst_ppl_u = exp(-1/length(D2_tst_loc)*sum(log(D2_pu(D2_tst_loc))));
disp(['Result : S2 Test Data Unigram Perplexity is ' num2str(D2_tst_ppl_u) ' for d = ' num2str(d)]);
disp(newline);

%% Bigram model N=2
%% modified unigram for bigram
a_w1w2 = sparse(D2_trn_ix_b(:,1),D2_trn_ix_b(:,2),D2_trn_b,Num_D2_vocab,Num_D2_vocab);
a_xw2 = sum(a_w1w2>0);
D2_trn_u_mod = full(a_xw2)';

% d=.2;
% D2(1) = d;
% D2(2) = d;
% D2(3) = d;
% d=.1:.1:.9;
% for i=1:length(d)
% D2(1) = d(i);
% D2(2) = d(i);
% D2(3) = d(i);
%
% D2_pu = max(D2_trn_u_mod - (D2_trn_u_mod==1)*D2(1)- (D2_trn_u_mod==2)*D2(2)- (D2_trn_u_mod>=3)*D2(3),0)/sum(D2_trn_u_mod);
% % D2_pu = D2_pu+(abs(1-sum(D2_pu))/Num_D2_vocab);
% D2_pu(end) = 1-sum(D2_pu);% <unk> prob
%
% % find dev perplexity
% D2_dev_ppl_u = exp(-1/length(D2_dev_loc)*sum(log(D2_pu(D2_dev_loc))));
% disp(['optim : S2 Dev Data Unigram Perplexity is ' num2str(D2_dev_ppl_u) ' for d = ' num2str(d(i))]);
%
% end

d=.5;
D2(1) = d;
D2(2) = d;
D2(3) = d;

D2_pu = max(D2_trn_u_mod - (D2_trn_u_mod==1)*D2(1)- (D2_trn_u_mod==2)*D2(2)- (D2_trn_u_mod>=3)*D2(3),0)/sum(D2_trn_u_mod);
% D2_pu = D2_pu+(abs(1-sum(D2_pu))/Num_D2_vocab);
D2_pu(end) = 1-sum(D2_pu);% <unk> prob

% % find dev perplexity
% D2_dev_ppl_u = exp(-1/length(D2_dev_loc)*sum(log(D2_pu(D2_dev_loc))));
% disp(['Result : S2 Dev Data Unigram Perplexity is ' num2str(D2_dev_ppl_u) ' for d = ' num2str(d)]);

%% Post Optimization Bigram model
% d=.1:.1:.9;
% for i=1:length(d)
% d2(1) = d(i);
% d2(2) = d(i);
% d2(3) = d(i);
% % d=.97;
% % d2(1) = d;
% % d2(2) = d;
% % d2(3) = d;
%
% D2_pb = max(D2_trn_b - (D2_trn_b==1)*d2(1)- (D2_trn_b==2)*d2(2)- (D2_trn_b>=3)*d2(3),0)./D2_trn_u(D2_trn_ix_b(:,1));
% a_w1x = sparse(D2_trn_ix_b(:,1),D2_trn_ix_b(:,2),D2_trn_b,Num_D2_vocab,Num_D2_vocab);
% b_w1 = full(sum(a_w1x==1,2)*d2(1)+ sum(a_w1x==2,2)*d2(2)+ sum(a_w1x==3,2)*d2(3));%./D2_trn_u(D2_trn_ix_b(:,2))
% D2_pb = D2_pb + b_w1(D2_trn_ix_b(:,1)).*D2_pu(D2_trn_ix_b(:,2))./D2_trn_u(D2_trn_ix_b(:,1));
% D2_pb_mat = sparse(D2_trn_ix_b(:,1),D2_trn_ix_b(:,2),D2_pb,Num_D2_vocab,Num_D2_vocab);
% D2_pb_norm = (1-sum(D2_pb_mat,2));
%
% [D2_pb_exists,D2_pb_exists_ix] = ismember(D2_dev_ix_b,D2_trn_ix_b,'rows');
% D2_pb_dev_ix = D2_pb_exists_ix(D2_pb_exists);
% sum_log_p = sum(log(D2_pb(D2_pb_dev_ix)))+sum(log(D2_pb_norm(D2_dev_ix_b(~D2_pb_exists,1)).*D2_pu(D2_dev_ix_b(~D2_pb_exists,2))));
% D2_dev_ppl_b = exp(-sum_log_p/length(D2_dev_ix_b));
% disp(['optim : S2 Dev Data Bigram Perplexity is ' num2str(D2_dev_ppl_b) ' for d = ' num2str(d(i))]);
%
% end

d=.8;
d2(1) = d;
d2(2) = d;
d2(3) = d;

D2_pb = max(D2_trn_b - (D2_trn_b==1)*d2(1)- (D2_trn_b==2)*d2(2)- (D2_trn_b>=3)*d2(3),0)./D2_trn_u(D2_trn_ix_b(:,1));
a_w1x = sparse(D2_trn_ix_b(:,1),D2_trn_ix_b(:,2),D2_trn_b,Num_D2_vocab,Num_D2_vocab);
b_w1 = full(sum(a_w1x==1,2)*d2(1)+ sum(a_w1x==2,2)*d2(2)+ sum(a_w1x==3,2)*d2(3));%./D2_trn_u(D2_trn_ix_b(:,2))
D2_pb = D2_pb + b_w1(D2_trn_ix_b(:,1)).*D2_pu(D2_trn_ix_b(:,2))./D2_trn_u(D2_trn_ix_b(:,1));
D2_pb_mat = sparse(D2_trn_ix_b(:,1),D2_trn_ix_b(:,2),D2_pb,Num_D2_vocab,Num_D2_vocab);
D2_pb_norm = (1-sum(D2_pb_mat,2));

[D2_pb_exists,D2_pb_exists_ix] = ismember(D2_dev_ix_b,D2_trn_ix_b,'rows');
D2_pb_dev_ix = D2_pb_exists_ix(D2_pb_exists);
sum_log_p = sum(log(D2_pb(D2_pb_dev_ix)))+sum(log(D2_pb_norm(D2_dev_ix_b(~D2_pb_exists,1)).*D2_pu(D2_dev_ix_b(~D2_pb_exists,2))));
D2_dev_ppl_b = exp(-sum_log_p/length(D2_dev_ix_b));
disp(['Result : S2 Dev Data Bigram Perplexity is ' num2str(D2_dev_ppl_b) ' for d = ' num2str(d)]);

[D2_pb_exists,D2_pb_exists_ix] = ismember(D2_tst_ix_b,D2_trn_ix_b,'rows');
D2_pb_tst_ix = D2_pb_exists_ix(D2_pb_exists);
sum_log_p = sum(log(D2_pb(D2_pb_tst_ix)))+sum(log(D2_pb_norm(D2_tst_ix_b(~D2_pb_exists,1)).*D2_pu(D2_tst_ix_b(~D2_pb_exists,2))));
D2_tst_ppl_b = exp(-sum_log_p/length(D2_tst_ix_b));
disp(['Result : S2 Test Data Bigram Perplexity is ' num2str(D2_tst_ppl_b) ' for d = ' num2str(d)]);
disp(newline);

%% Bigram Sentence generation
w1=[];
str ='';%<\s\>
for i=1:9
    num_toD2_pwords = randi(10);
    if isempty(w1)
        % [~,w1_ix] = mink(D2_pu(1:end-1),10000);
        % w1 = w1_ix(randi(length(w1_ix)));
        w1 = randi(Num_D2_vocab-1);
        % w1 = w1_ix(end);
        str = D2_vocab{w1};
    end
    w1_ix = find(D2_trn_ix_b(:,1)==w1);
    [max_D2_pb,w2_ix] = maxk(D2_pb(w1_ix),num_toD2_pwords);% D2_trn_ix_b(w1_ix,2)
    rand_toD2_pk = randi(length(w2_ix));
    max_D2_pb = max_D2_pb(rand_toD2_pk);
    [max_D2_pb_n,norm_ix] = maxk(D2_pb_norm(w1)*D2_pu(1:end-1),num_toD2_pwords);% D2_trn_ix_b(w1_ix,2)
    max_D2_pb_n = max_D2_pb_n(randi(length(norm_ix)));
    if max_D2_pb > max_D2_pb_n
        w2 = D2_trn_ix_b(w1_ix(w2_ix(rand_toD2_pk)),2);
    else
        w2 = norm_ix(randi(length(norm_ix)));
    end
    str = [str ' ' D2_vocab{w2}];
    w1 = w2;
end
disp(['Sentence generated from S2 Setting - "' str ' "'])
disp(newline);

end

%% S3: Train: D1-Train + D2-Train, Test: D1-Test
function setting_s3
global S3_trn_ix_b S3_dev_ix_b S3_tst_ix_b S3_trn_u S3_trn_b S3_dev_b S3_tst_b S3_pu S3_pb S3_trn_loc S3_dev_loc S3_tst_loc S3_vocab Num_S3_vocab
global  D1_trn D1_dev D1_tst D2_trn D2_dev

%% Compute the n-gram counts fron the Datafiles
if isempty(S3_trn_u)||isempty(S3_trn_b)% clear
    %% Unigram - trn
    S3_trn = [D1_trn D2_trn];
    S3_dev = [D1_dev D2_dev];
    S3_tst = D1_tst;
    
    S3_trn_u_tkn = regexp(S3_trn,'\S+','match');%( )?
    S3_vocab = [unique(S3_trn_u_tkn,'stable') {'<UNK>'}];% adding extra Unknown word to Vocab
    Num_S3_vocab = length(S3_vocab);
    [~,S3_trn_loc] = ismember(S3_trn_u_tkn,S3_vocab);
    S3_trn_loc_str = regexprep(num2str(S3_trn_loc),'\s+',' ');
    space_idx = regexp(S3_trn_loc_str,' ');
    S3_trn_u = hist(S3_trn_loc,1:Num_S3_vocab)';
    
    %% Bigram - trn
    S3_2a_tkn = regexp(S3_trn_loc_str,'\d+ \d+','match');
    S3_2b_tkn = regexp(S3_trn_loc_str(space_idx(1)+1:end),'\d+ \d+','match');% as overlapping find is not
    S3_trn_b_tkn = [S3_2a_tkn S3_2b_tkn];
    vocab2 = unique(S3_trn_b_tkn,'stable');
    Num_vocab2 = length(vocab2);
    [~,vocab2_loc] = ismember(S3_trn_b_tkn,vocab2);
    S3_trn_b = hist(vocab2_loc,1:Num_vocab2)';
    S3_trn_ix_b = cell2mat(cellfun(@str2num,vocab2','UniformOutput',false));
    
    %     %% Trigram - trn
    %     S3_3a_tkn = regexp(S3_trn_loc_str,'\d+ \d+ \d+','match');
    %     S3_3b_tkn = regexp(S3_trn_loc_str(space_idx(1)+1:end),'\d+ \d+ \d+','match');% as overlapping find is not
    %     S3_3c_tkn = regexp(S3_trn_loc_str(space_idx(2)+1:end),'\d+ \d+ \d+','match');% as overlapping find is not
    %     S3_trn_t_tkn = [S3_3a_tkn S3_3b_tkn S3_3c_tkn];
    %     vocab3 = unique(S3_trn_t_tkn,'stable');
    %     Num_vocab3 = length(vocab3);
    %     [~,vocab3_loc] = ismember(S3_trn_t_tkn,vocab3);
    %     S3_trn_t = hist(vocab3_loc,1:Num_vocab3)';
    %     S3_trn_ix_t = cell2mat(cellfun(@str2num,vocab3','UniformOutput',false));
    
    %% Unigram - dev
    S3_dev_u_tkn = regexp(S3_dev,'\S+','match');%( )?
    [~,S3_dev_loc] = ismember(S3_dev_u_tkn,S3_vocab);
    S3_dev_loc(S3_dev_loc==0)=Num_S3_vocab;% Unknown
    S3_dev_loc_str = regexprep(num2str(S3_dev_loc),'\s+',' ');
    space_idx = regexp(S3_dev_loc_str,' ');
    
    %% Bigram - dev
    S3_2a_tkn = regexp(S3_dev_loc_str,'\d+ \d+','match');
    S3_2b_tkn = regexp(S3_dev_loc_str(space_idx(1)+1:end),'\d+ \d+','match');% as overlapping find is not
    S3_dev_2_tkn = [S3_2a_tkn S3_2b_tkn];
    [~,vocab2_loc] = ismember(S3_dev_2_tkn,vocab2);
    S3_dev_b = hist(vocab2_loc,1:Num_vocab2)';
    S3_dev_ix_b = cell2mat(cellfun(@str2num,S3_dev_2_tkn','UniformOutput',false));
    
    %     %% Trigram - dev
    %     S3_3a_tkn = regexp(S3_dev_loc_str,'\d+ \d+ \d+','match');
    %     S3_3b_tkn = regexp(S3_dev_loc_str(space_idx(1)+1:end),'\d+ \d+ \d+','match');% as overlapping find is not
    %     S3_3c_tkn = regexp(S3_dev_loc_str(space_idx(2)+1:end),'\d+ \d+ \d+','match');% as overlapping find is not
    %     S3_dev_t_tkn = [S3_3a_tkn S3_3b_tkn S3_3c_tkn];
    %     [~,vocab3_loc] = ismember(S3_dev_t_tkn,vocab3);
    %     S3_dev_t = hist(vocab3_loc,1:Num_vocab3)';
    %     S3_dev_ix_t = cell2mat(cellfun(@str2num,S3_dev_t_tkn','UniformOutput',false));
    
    %% Unigram - tst
    S3_tst_u_tkn = regexp(S3_tst,'\S+','match');%( )?
    [~,S3_tst_loc] = ismember(S3_tst_u_tkn,S3_vocab);
    S3_tst_loc(S3_tst_loc==0)=Num_S3_vocab;% Unknown
    S3_tst_loc_str = regexprep(num2str(S3_tst_loc),'\s+',' ');
    space_idx = regexp(S3_tst_loc_str,' ');
    
    %% Bigram -tst
    S3_2a_tkn = regexp(S3_tst_loc_str,'\d+ \d+','match');
    S3_2b_tkn = regexp(S3_tst_loc_str(space_idx(1)+1:end),'\d+ \d+','match');% as overlapping find is not
    S3_tst_b_tkn = [S3_2a_tkn S3_2b_tkn];
    [~,vocab2_loc] = ismember(S3_tst_b_tkn,vocab2);
    S3_tst_b = hist(vocab2_loc,1:Num_vocab2)';
    S3_tst_ix_b = cell2mat(cellfun(@str2num,S3_tst_b_tkn','UniformOutput',false));
    
end

%     %% Trigram - tst
%     S3_3a_tkn = regexp(S3_tst_loc_str,'\d+ \d+ \d+','match');
%     S3_3b_tkn = regexp(S3_tst_loc_str(space_idx(1)+1:end),'\d+ \d+ \d+','match');% as overlapping find is not
%     S3_3c_tkn = regexp(S3_tst_loc_str(space_idx(2)+1:end),'\d+ \d+ \d+','match');% as overlapping find is not
%     S3_tst_t_tkn = [S3_3a_tkn S3_3b_tkn S3_3c_tkn];
%     [~,vocab3_loc] = ismember(S3_tst_t_tkn,vocab3);
%     S3_tst_t = hist(vocab3_loc,1:Num_vocab3)';
%     S3_tst_ix_t = cell2mat(cellfun(@str2num,S3_tst_t_tkn','UniformOutput',false));


%% Laplace Smoothing
% new_C1 = C1+1;
% S3_pw1 = log(new_C1/sum(new_C1));
% Observation :
% since vocabulary is very large, Add one smoothing (Laplace smoothing)
% does not help here as adding 1 to numerator has negligible effect when
% dividing by Num_S3_vocab = 44674. Still the prob is very low. Infact we have
% even made it worse for probabilities good counts earlier.
% Even Add k smoothing also produces same result because of same large
% Num_Vocab = 44674 words
% So We check the criteria for using Good-Turing Smoothing.

%% Good-Turing Smoothing
% [max_C1,max_C1_idx] = max(S3_trn_C1);
% % C1_GT=C1;
% % C1_GT(C1>=3)=3;
% % max_C1 = 3;
% Num_C1 = hist(S3_trn_C1,1:max_C1);
% zero_Num_C1_idx = find(Num_C1==0);
% % Observation
% % since max_C1 = 42603(the), is very large, there are lots of C
% % which have zero Num_C "zero_Num_C1_idx". So Good-Turing Smoothing cannot be used here.
% % So we move to Kneser-Ney smoothing

%% We use modified Kneser-Ney
%% Unigram model N=1
% optimization for d in unigram using held out set
Num_S3_vocab = length(S3_vocab);
% d=.1:.1:.9;
% for i=1:length(d)
%     d1(1) = d(i);
%     d1(2) = d(i);
%     d1(3) = d(i);
%     
%     S3_pu = max(S3_trn_u - (S3_trn_u==1)*d1(1)- (S3_trn_u==2)*d1(2)- (S3_trn_u>=3)*d1(3),0)/sum(S3_trn_u);
%     % S3_pu = S3_pu+(abs(1-sum(S3_pu))/Num_S3_vocab);
%     S3_pu(end) = 1-sum(S3_pu);% <unk> prob
%     
%     % find dev perplexity
%     S3_dev_ppl_u = exp(-1/length(S3_dev_loc)*sum(log(S3_pu(S3_dev_loc))));
%     disp(['optim : S3 Dev Data Unigram Perplexity is ' num2str(S3_dev_ppl_u) ' for d = ' num2str(d(i))]);
%     
% end

%% Post optimization unigram model
d=.5;
d1(1) = d;
d1(2) = d;
d1(3) = d;

S3_pu = max(S3_trn_u - (S3_trn_u==1)*d1(1)- (S3_trn_u==2)*d1(2)- (S3_trn_u>=3)*d1(3),0)/sum(S3_trn_u);
% S3_pu = S3_pu+(abs(1-sum(S3_pu))/Num_S3_vocab);
S3_pu(end) = 1-sum(S3_pu);% <unk> prob

% find dev perplexity
S3_dev_ppl_u = exp(-1/length(S3_dev_loc)*sum(log(S3_pu(S3_dev_loc))));
disp(['Result : S3 Dev Data Unigram Perplexity is ' num2str(S3_dev_ppl_u) ' for d = ' num2str(d)]);


% test perplexity Unigram
S3_tst_ppl_u = exp(-1/length(S3_tst_loc)*sum(log(S3_pu(S3_tst_loc))));
disp(['Result : S3 Test Data Unigram Perplexity is ' num2str(S3_tst_ppl_u) ' for d = ' num2str(d)]);
disp(newline);

%% Bigram model N=2
%% modified unigram for bigram
a_w1w2 = sparse(S3_trn_ix_b(:,1),S3_trn_ix_b(:,2),S3_trn_b,Num_S3_vocab,Num_S3_vocab);
a_xw2 = sum(a_w1w2>0);
S3_trn_u_mod = full(a_xw2)';

% d=.1:.1:.9;
% for i=1:length(d)
% d1(1) = d(i);
% d1(2) = d(i);
% d1(3) = d(i);
% 
% S3_pu = max(S3_trn_u_mod - (S3_trn_u_mod==1)*d1(1)- (S3_trn_u_mod==2)*d1(2)- (S3_trn_u_mod>=3)*d1(3),0)/sum(S3_trn_u_mod);
% % S3_pu = S3_pu+(abs(1-sum(S3_pu))/Num_S3_vocab);
% S3_pu(end) = 1-sum(S3_pu);% <unk> prob
% 
% S3_dev_ppl_u = exp(-1/length(S3_dev_loc)*sum(log(S3_pu(S3_dev_loc))));
% disp(['optim : S3 Dev Data Unigram Perplexity is ' num2str(S3_dev_ppl_u) ' for d = ' num2str(d(i))]);
% 
% end

d=.4;
d1(1) = d;
d1(2) = d;
d1(3) = d;

S3_pu = max(S3_trn_u_mod - (S3_trn_u_mod==1)*d1(1)- (S3_trn_u_mod==2)*d1(2)- (S3_trn_u_mod>=3)*d1(3),0)/sum(S3_trn_u_mod);
% S3_pu = S3_pu+(abs(1-sum(S3_pu))/Num_S3_vocab);
S3_pu(end) = 1-sum(S3_pu);% <unk> prob

%% Post Optimization Bigram model
% d=.1:.1:.9;
% for i=1:length(d)
% d2(1) = d(i);
% d2(2) = d(i);
% d2(3) = d(i);
% % d=.8;
% % d2(1) = d;
% % d2(2) = d;
% % d2(3) = d;
% 
% S3_pb = max(S3_trn_b - (S3_trn_b==1)*d2(1)- (S3_trn_b==2)*d2(2)- (S3_trn_b>=3)*d2(3),0)./S3_trn_u(S3_trn_ix_b(:,1));
% a_w1x = sparse(S3_trn_ix_b(:,1),S3_trn_ix_b(:,2),S3_trn_b,Num_S3_vocab,Num_S3_vocab);
% b_w1 = full(sum(a_w1x==1,2)*d2(1)+ sum(a_w1x==2,2)*d2(2)+ sum(a_w1x==3,2)*d2(3));%./S3_trn_u(S3_trn_ix_b(:,2))
% S3_pb = S3_pb + b_w1(S3_trn_ix_b(:,1)).*S3_pu(S3_trn_ix_b(:,2))./S3_trn_u(S3_trn_ix_b(:,1));
% S3_pb_mat = sparse(S3_trn_ix_b(:,1),S3_trn_ix_b(:,2),S3_pb,Num_S3_vocab,Num_S3_vocab);
% S3_pb_norm = (1-sum(S3_pb_mat,2));
% 
% [S3_pb_exists,S3_pb_exists_ix] = ismember(S3_dev_ix_b,S3_trn_ix_b,'rows');
% S3_pb_dev_ix = S3_pb_exists_ix(S3_pb_exists);
% sum_log_p = sum(log(S3_pb(S3_pb_dev_ix)))+sum(log(S3_pb_norm(S3_dev_ix_b(~S3_pb_exists,1)).*S3_pu(S3_dev_ix_b(~S3_pb_exists,2))));
% S3_dev_ppl_b = exp(-sum_log_p/length(S3_dev_ix_b));
% disp(['optim : S3 Dev Data Bigram Perplexity is ' num2str(S3_dev_ppl_b) ' for d = ' num2str(d(i))]);
% 
% end

d=.8;
d2(1) = d;
d2(2) = d;
d2(3) = d;

S3_pb = max(S3_trn_b - (S3_trn_b==1)*d2(1)- (S3_trn_b==2)*d2(2)- (S3_trn_b>=3)*d2(3),0)./S3_trn_u(S3_trn_ix_b(:,1));
a_w1x = sparse(S3_trn_ix_b(:,1),S3_trn_ix_b(:,2),S3_trn_b,Num_S3_vocab,Num_S3_vocab);
b_w1 = full(sum(a_w1x==1,2)*d2(1)+ sum(a_w1x==2,2)*d2(2)+ sum(a_w1x==3,2)*d2(3));%./S3_trn_u(S3_trn_ix_b(:,2))
S3_pb = S3_pb + b_w1(S3_trn_ix_b(:,1)).*S3_pu(S3_trn_ix_b(:,2))./S3_trn_u(S3_trn_ix_b(:,1));
S3_pb_mat = sparse(S3_trn_ix_b(:,1),S3_trn_ix_b(:,2),S3_pb,Num_S3_vocab,Num_S3_vocab);
S3_pb_norm = (1-sum(S3_pb_mat,2));

[S3_pb_exists,S3_pb_exists_ix] = ismember(S3_dev_ix_b,S3_trn_ix_b,'rows');
S3_pb_dev_ix = S3_pb_exists_ix(S3_pb_exists);
sum_log_p = sum(log(S3_pb(S3_pb_dev_ix)))+sum(log(S3_pb_norm(S3_dev_ix_b(~S3_pb_exists,1)).*S3_pu(S3_dev_ix_b(~S3_pb_exists,2))));
S3_dev_ppl_b = exp(-sum_log_p/length(S3_dev_ix_b));
disp(['Result : S3 Dev Data Bigram Perplexity is ' num2str(S3_dev_ppl_b) ' for d = ' num2str(d)]);

[S3_pb_exists,S3_pb_exists_ix] = ismember(S3_tst_ix_b,S3_trn_ix_b,'rows');
S3_pb_tst_ix = S3_pb_exists_ix(S3_pb_exists);
sum_log_p = sum(log(S3_pb(S3_pb_tst_ix)))+sum(log(S3_pb_norm(S3_tst_ix_b(~S3_pb_exists,1)).*S3_pu(S3_tst_ix_b(~S3_pb_exists,2))));
S3_tst_ppl_b = exp(-sum_log_p/length(S3_tst_ix_b));
disp(['Result : S3 Test Data Bigram Perplexity is ' num2str(S3_tst_ppl_b) ' for d = ' num2str(d)]);
disp(newline);

%% Bigram Sentence generation
w1=[];
str ='';%<\s\>
for i=1:9
    num_toS3_pwords = randi(10);
    if isempty(w1)
        % [~,w1_ix] = mink(S3_pu(1:end-1),10000);
        % w1 = w1_ix(randi(length(w1_ix)));
        w1 = randi(Num_S3_vocab-1);
        % w1 = w1_ix(end);
        str = S3_vocab{w1};
    end
    w1_ix = find(S3_trn_ix_b(:,1)==w1);
    [max_S3_pb,w2_ix] = maxk(S3_pb(w1_ix),num_toS3_pwords);% S3_trn_ix_b(w1_ix,2)
    rand_toS3_pk = randi(length(w2_ix));
    max_S3_pb = max_S3_pb(rand_toS3_pk);
    [max_S3_pb_n,norm_ix] = maxk(S3_pb_norm(w1)*S3_pu(1:end-1),num_toS3_pwords);% S3_trn_ix_b(w1_ix,2)
    max_S3_pb_n = max_S3_pb_n(randi(length(norm_ix)));
    if max_S3_pb > max_S3_pb_n
        w2 = S3_trn_ix_b(w1_ix(w2_ix(rand_toS3_pk)),2);
    else
        w2 = norm_ix(randi(length(norm_ix)));
    end
    str = [str ' ' S3_vocab{w2}];
    w1 = w2;
end
disp(['Sentence generated from S3 Setting - "' str ' "'])
disp(newline);

end

%% S4: Train: D1-Train + D2-Train, Test: D2-Test
function setting_s4
global S4_trn_ix_b S4_dev_ix_b S4_tst_ix_b S4_trn_u S4_trn_b S4_dev_b S4_tst_b S4_pu S4_pb S4_trn_loc S4_dev_loc S4_tst_loc S4_vocab Num_S4_vocab
global  D1_trn D1_dev D2_trn D2_dev D2_tst

%% Compute the n-gram counts fron the Datafiles
if isempty(S4_trn_u)||isempty(S4_trn_b)% clear
%% Unigram - trn
S4_trn = [D1_trn D2_trn];
S4_dev = [D1_dev D2_dev];
S4_tst = D2_tst;

S4_trn_u_tkn = regexp(S4_trn,'\S+','match');%( )?
S4_vocab = [unique(S4_trn_u_tkn,'stable') {'<UNK>'}];% adding extra Unknown word to Vocab
Num_S4_vocab = length(S4_vocab);
[~,S4_trn_loc] = ismember(S4_trn_u_tkn,S4_vocab);
S4_trn_loc_str = regexprep(num2str(S4_trn_loc),'\s+',' ');
space_idx = regexp(S4_trn_loc_str,' ');
S4_trn_u = hist(S4_trn_loc,1:Num_S4_vocab)';

%% Bigram - trn
S4_2a_tkn = regexp(S4_trn_loc_str,'\d+ \d+','match');
S4_2b_tkn = regexp(S4_trn_loc_str(space_idx(1)+1:end),'\d+ \d+','match');% as overlapping find is not
S4_trn_b_tkn = [S4_2a_tkn S4_2b_tkn];
vocab2 = unique(S4_trn_b_tkn,'stable');
Num_vocab2 = length(vocab2);
[~,vocab2_loc] = ismember(S4_trn_b_tkn,vocab2);
S4_trn_b = hist(vocab2_loc,1:Num_vocab2)';
S4_trn_ix_b = cell2mat(cellfun(@str2num,vocab2','UniformOutput',false));

%     %% Trigram - trn
%     S4_3a_tkn = regexp(S4_trn_loc_str,'\d+ \d+ \d+','match');
%     S4_3b_tkn = regexp(S4_trn_loc_str(space_idx(1)+1:end),'\d+ \d+ \d+','match');% as overlapping find is not
%     S4_3c_tkn = regexp(S4_trn_loc_str(space_idx(2)+1:end),'\d+ \d+ \d+','match');% as overlapping find is not
%     S4_trn_t_tkn = [S4_3a_tkn S4_3b_tkn S4_3c_tkn];
%     vocab3 = unique(S4_trn_t_tkn,'stable');
%     Num_vocab3 = length(vocab3);
%     [~,vocab3_loc] = ismember(S4_trn_t_tkn,vocab3);
%     S4_trn_t = hist(vocab3_loc,1:Num_vocab3)';
%     S4_trn_ix_t = cell2mat(cellfun(@str2num,vocab3','UniformOutput',false));

%% Unigram - dev
S4_dev_u_tkn = regexp(S4_dev,'\S+','match');%( )?
[~,S4_dev_loc] = ismember(S4_dev_u_tkn,S4_vocab);
S4_dev_loc(S4_dev_loc==0)=Num_S4_vocab;% Unknown
S4_dev_loc_str = regexprep(num2str(S4_dev_loc),'\s+',' ');
space_idx = regexp(S4_dev_loc_str,' ');

%% Bigram - dev
S4_2a_tkn = regexp(S4_dev_loc_str,'\d+ \d+','match');
S4_2b_tkn = regexp(S4_dev_loc_str(space_idx(1)+1:end),'\d+ \d+','match');% as overlapping find is not
S4_dev_2_tkn = [S4_2a_tkn S4_2b_tkn];
[~,vocab2_loc] = ismember(S4_dev_2_tkn,vocab2);
S4_dev_b = hist(vocab2_loc,1:Num_vocab2)';
S4_dev_ix_b = cell2mat(cellfun(@str2num,S4_dev_2_tkn','UniformOutput',false));

%     %% Trigram - dev
%     S4_3a_tkn = regexp(S4_dev_loc_str,'\d+ \d+ \d+','match');
%     S4_3b_tkn = regexp(S4_dev_loc_str(space_idx(1)+1:end),'\d+ \d+ \d+','match');% as overlapping find is not
%     S4_3c_tkn = regexp(S4_dev_loc_str(space_idx(2)+1:end),'\d+ \d+ \d+','match');% as overlapping find is not
%     S4_dev_t_tkn = [S4_3a_tkn S4_3b_tkn S4_3c_tkn];
%     [~,vocab3_loc] = ismember(S4_dev_t_tkn,vocab3);
%     S4_dev_t = hist(vocab3_loc,1:Num_vocab3)';
%     S4_dev_ix_t = cell2mat(cellfun(@str2num,S4_dev_t_tkn','UniformOutput',false));

%% Unigram - tst
S4_tst_u_tkn = regexp(S4_tst,'\S+','match');%( )?
[~,S4_tst_loc] = ismember(S4_tst_u_tkn,S4_vocab);
S4_tst_loc(S4_tst_loc==0)=Num_S4_vocab;% Unknown
S4_tst_loc_str = regexprep(num2str(S4_tst_loc),'\s+',' ');
space_idx = regexp(S4_tst_loc_str,' ');

%% Bigram -tst
S4_2a_tkn = regexp(S4_tst_loc_str,'\d+ \d+','match');
S4_2b_tkn = regexp(S4_tst_loc_str(space_idx(1)+1:end),'\d+ \d+','match');% as overlapping find is not
S4_tst_b_tkn = [S4_2a_tkn S4_2b_tkn];
[~,vocab2_loc] = ismember(S4_tst_b_tkn,vocab2);
S4_tst_b = hist(vocab2_loc,1:Num_vocab2)';
S4_tst_ix_b = cell2mat(cellfun(@str2num,S4_tst_b_tkn','UniformOutput',false));

%     %% Trigram - tst
%     S4_3a_tkn = regexp(S4_tst_loc_str,'\d+ \d+ \d+','match');
%     S4_3b_tkn = regexp(S4_tst_loc_str(space_idx(1)+1:end),'\d+ \d+ \d+','match');% as overlapping find is not
%     S4_3c_tkn = regexp(S4_tst_loc_str(space_idx(2)+1:end),'\d+ \d+ \d+','match');% as overlapping find is not
%     S4_tst_t_tkn = [S4_3a_tkn S4_3b_tkn S4_3c_tkn];
%     [~,vocab3_loc] = ismember(S4_tst_t_tkn,vocab3);
%     S4_tst_t = hist(vocab3_loc,1:Num_vocab3)';
%     S4_tst_ix_t = cell2mat(cellfun(@str2num,S4_tst_t_tkn','UniformOutput',false));
end

%% Laplace Smoothing
% new_C1 = C1+1;
% S4_pw1 = log(new_C1/sum(new_C1));
% Observation :
% since vocabulary is very large, Add one smoothing (Laplace smoothing)
% does not help here as adding 1 to numerator has negligible effect when
% dividing by Num_S4_vocab = 44674. Still the prob is very low. Infact we have
% even made it worse for probabilities good counts earlier.
% Even Add k smoothing also produces same result because of same large
% Num_Vocab = 44674 words
% So We check the criteria for using Good-Turing Smoothing.

%% Good-Turing Smoothing
% [max_C1,max_C1_idx] = max(S4_trn_C1);
% % C1_GT=C1;
% % C1_GT(C1>=3)=3;
% % max_C1 = 3;
% Num_C1 = hist(S4_trn_C1,1:max_C1);
% zero_Num_C1_idx = find(Num_C1==0);
% % Observation
% % since max_C1 = 42603(the), is very large, there are lots of C
% % which have zero Num_C "zero_Num_C1_idx". So Good-Turing Smoothing cannot be used here.
% % So we move to Kneser-Ney smoothing

%% We use modified Kneser-Ney
%% Unigram model N=1
% optimization for d in unigram using held out set
Num_S4_vocab = length(S4_vocab);
% d=.1:.1:.9;
% for i=1:length(d)
% d1(1) = d(i);
% d1(2) = d(i);
% d1(3) = d(i);
% 
% S4_pu = max(S4_trn_u - (S4_trn_u==1)*d1(1)- (S4_trn_u==2)*d1(2)- (S4_trn_u>=3)*d1(3),0)/sum(S4_trn_u);
% % S4_pu = S4_pu+(abs(1-sum(S4_pu))/Num_S4_vocab);
% S4_pu(end) = 1-sum(S4_pu);% <unk> prob
% 
% % find dev perplexity
% S4_dev_ppl_u = exp(-1/length(S4_dev_loc)*sum(log(S4_pu(S4_dev_loc))));
% disp(['optim : S4 Dev Data Unigram Perplexity is ' num2str(S4_dev_ppl_u) ' for d = ' num2str(d(i))]);
% 
% end

%% Post optimization unigram model
d=.5;
d1(1) = d;
d1(2) = d;
d1(3) = d;

S4_pu = max(S4_trn_u - (S4_trn_u==1)*d1(1)- (S4_trn_u==2)*d1(2)- (S4_trn_u>=3)*d1(3),0)/sum(S4_trn_u);
% S4_pu = S4_pu+(abs(1-sum(S4_pu))/Num_S4_vocab);
S4_pu(end) = 1-sum(S4_pu);% <unk> prob

% find dev perplexity
S4_dev_ppl_u = exp(-1/length(S4_dev_loc)*sum(log(S4_pu(S4_dev_loc))));
disp(['Result : S4 Dev Data Unigram Perplexity is ' num2str(S4_dev_ppl_u) ' for d = ' num2str(d)]);


% test perplexity Unigram
S4_tst_ppl_u = exp(-1/length(S4_tst_loc)*sum(log(S4_pu(S4_tst_loc))));
disp(['Result : S4 Test Data Unigram Perplexity is ' num2str(S4_tst_ppl_u) ' for d = ' num2str(d)]);
disp(newline);

%% Bigram model N=2
%% modified unigram for bigram
a_w1w2 = sparse(S4_trn_ix_b(:,1),S4_trn_ix_b(:,2),S4_trn_b,Num_S4_vocab,Num_S4_vocab);
a_xw2 = sum(a_w1w2>0);
S4_trn_u_mod = full(a_xw2)';

% d=.1:.1:.9;
% for i=1:length(d)
% d1(1) = d(i);
% d1(2) = d(i);
% d1(3) = d(i);
% % d=.2;
% % d1(1) = d;
% % d1(2) = d;
% % d1(3) = d;
% 
% S4_pu = max(S4_trn_u_mod - (S4_trn_u_mod==1)*d1(1)- (S4_trn_u_mod==2)*d1(2)- (S4_trn_u_mod>=3)*d1(3),0)/sum(S4_trn_u_mod);
% % S4_pu = S4_pu+(abs(1-sum(S4_pu))/Num_S4_vocab);
% S4_pu(end) = 1-sum(S4_pu);% <unk> prob
% 
% S4_dev_ppl_u = exp(-1/length(S4_dev_loc)*sum(log(S4_pu(S4_dev_loc))));
% disp(['optim : S4 Dev Data Unigram Perplexity is ' num2str(S4_dev_ppl_u) ' for d = ' num2str(d(i))]);
% 
% end

d=.4;
d1(1) = d;
d1(2) = d;
d1(3) = d;

S4_pu = max(S4_trn_u_mod - (S4_trn_u_mod==1)*d1(1)- (S4_trn_u_mod==2)*d1(2)- (S4_trn_u_mod>=3)*d1(3),0)/sum(S4_trn_u_mod);
% S4_pu = S4_pu+(abs(1-sum(S4_pu))/Num_S4_vocab);
S4_pu(end) = 1-sum(S4_pu);% <unk> prob

% %% Post Optimization Bigram model
% d=.1:.1:.9;
% for i=1:length(d)
% d2(1) = d(i);
% d2(2) = d(i);
% d2(3) = d(i);
% % d=.8;
% % d2(1) = d;
% % d2(2) = d;
% % d2(3) = d;
% 
% S4_pb = max(S4_trn_b - (S4_trn_b==1)*d2(1)- (S4_trn_b==2)*d2(2)- (S4_trn_b>=3)*d2(3),0)./S4_trn_u(S4_trn_ix_b(:,1));
% a_w1x = sparse(S4_trn_ix_b(:,1),S4_trn_ix_b(:,2),S4_trn_b,Num_S4_vocab,Num_S4_vocab);
% b_w1 = full(sum(a_w1x==1,2)*d2(1)+ sum(a_w1x==2,2)*d2(2)+ sum(a_w1x==3,2)*d2(3));%./S4_trn_u(S4_trn_ix_b(:,2))
% S4_pb = S4_pb + b_w1(S4_trn_ix_b(:,1)).*S4_pu(S4_trn_ix_b(:,2))./S4_trn_u(S4_trn_ix_b(:,1));
% S4_pb_mat = sparse(S4_trn_ix_b(:,1),S4_trn_ix_b(:,2),S4_pb,Num_S4_vocab,Num_S4_vocab);
% S4_pb_norm = (1-sum(S4_pb_mat,2));
% 
% [S4_pb_exists,S4_pb_exists_ix] = ismember(S4_dev_ix_b,S4_trn_ix_b,'rows');
% S4_pb_dev_ix = S4_pb_exists_ix(S4_pb_exists);
% sum_log_p = sum(log(S4_pb(S4_pb_dev_ix)))+sum(log(S4_pb_norm(S4_dev_ix_b(~S4_pb_exists,1)).*S4_pu(S4_dev_ix_b(~S4_pb_exists,2))));
% S4_dev_ppl_b = exp(-sum_log_p/length(S4_dev_ix_b));
% disp(['optim : S4 Dev Data Bigram Perplexity is ' num2str(S4_dev_ppl_b) ' for d = ' num2str(d(i))]);
% 
% end

d=.8;
d2(1) = d;
d2(2) = d;
d2(3) = d;

S4_pb = max(S4_trn_b - (S4_trn_b==1)*d2(1)- (S4_trn_b==2)*d2(2)- (S4_trn_b>=3)*d2(3),0)./S4_trn_u(S4_trn_ix_b(:,1));
a_w1x = sparse(S4_trn_ix_b(:,1),S4_trn_ix_b(:,2),S4_trn_b,Num_S4_vocab,Num_S4_vocab);
b_w1 = full(sum(a_w1x==1,2)*d2(1)+ sum(a_w1x==2,2)*d2(2)+ sum(a_w1x==3,2)*d2(3));%./S4_trn_u(S4_trn_ix_b(:,2))
S4_pb = S4_pb + b_w1(S4_trn_ix_b(:,1)).*S4_pu(S4_trn_ix_b(:,2))./S4_trn_u(S4_trn_ix_b(:,1));
S4_pb_mat = sparse(S4_trn_ix_b(:,1),S4_trn_ix_b(:,2),S4_pb,Num_S4_vocab,Num_S4_vocab);
S4_pb_norm = (1-sum(S4_pb_mat,2));

[S4_pb_exists,S4_pb_exists_ix] = ismember(S4_dev_ix_b,S4_trn_ix_b,'rows');
S4_pb_dev_ix = S4_pb_exists_ix(S4_pb_exists);
sum_log_p = sum(log(S4_pb(S4_pb_dev_ix)))+sum(log(S4_pb_norm(S4_dev_ix_b(~S4_pb_exists,1)).*S4_pu(S4_dev_ix_b(~S4_pb_exists,2))));
S4_dev_ppl_b = exp(-sum_log_p/length(S4_dev_ix_b));
disp(['Result : S4 Dev Data Bigram Perplexity is ' num2str(S4_dev_ppl_b) ' for d = ' num2str(d)]);

[S4_pb_exists,S4_pb_exists_ix] = ismember(S4_tst_ix_b,S4_trn_ix_b,'rows');
S4_pb_tst_ix = S4_pb_exists_ix(S4_pb_exists);
sum_log_p = sum(log(S4_pb(S4_pb_tst_ix)))+sum(log(S4_pb_norm(S4_tst_ix_b(~S4_pb_exists,1)).*S4_pu(S4_tst_ix_b(~S4_pb_exists,2))));
S4_tst_ppl_b = exp(-sum_log_p/length(S4_tst_ix_b));
disp(['Result : S4 Test Data Bigram Perplexity is ' num2str(S4_tst_ppl_b) ' for d = ' num2str(d)]);
disp(newline);

%% Bigram Sentence generation
w1=[];
str ='';%<\s\>
for i=1:9
    num_toS4_pwords = randi(10);
    if isempty(w1)
        % [~,w1_ix] = mink(S4_pu(1:end-1),10000);
        % w1 = w1_ix(randi(length(w1_ix)));
        w1 = randi(Num_S4_vocab-1);
        % w1 = w1_ix(end);
        str = S4_vocab{w1};
    end
    w1_ix = find(S4_trn_ix_b(:,1)==w1);
    [max_S4_pb,w2_ix] = maxk(S4_pb(w1_ix),num_toS4_pwords);% S4_trn_ix_b(w1_ix,2)
    rand_toS4_pk = randi(length(w2_ix));
    max_S4_pb = max_S4_pb(rand_toS4_pk);
    [max_S4_pb_n,norm_ix] = maxk(S4_pb_norm(w1)*S4_pu(1:end-1),num_toS4_pwords);% S4_trn_ix_b(w1_ix,2)
    max_S4_pb_n = max_S4_pb_n(randi(length(norm_ix)));
    if max_S4_pb > max_S4_pb_n
        w2 = S4_trn_ix_b(w1_ix(w2_ix(rand_toS4_pk)),2);
    else
        w2 = norm_ix(randi(length(norm_ix)));
    end
    str = [str ' ' S4_vocab{w2}];
    w1 = w2;
end
disp(['Sentence generated from S4 Setting - "' str ' "'])
disp(newline);

end
