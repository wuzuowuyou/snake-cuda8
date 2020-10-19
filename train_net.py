from lib.config import cfg, args
from lib.networks import make_network
from lib.train import make_trainer, make_optimizer, make_lr_scheduler, make_recorder, set_lr_scheduler
from lib.datasets import make_data_loader
from lib.utils.net_utils import load_model, save_model, load_network
from lib.evaluators import make_evaluator
import torch.multiprocessing

torch.backends.cudnn.enabled = False


def train(cfg, network):
    trainer = make_trainer(cfg, network)
    optimizer = make_optimizer(cfg, network)
    scheduler = make_lr_scheduler(cfg, optimizer)
    recorder = make_recorder(cfg)
    evaluator = make_evaluator(cfg)

    begin_epoch = load_model(network, optimizer, scheduler, recorder, cfg.model_dir, resume=cfg.resume)
    # set_lr_scheduler(cfg, scheduler)

    train_loader = make_data_loader(cfg, is_train=True)
    val_loader = make_data_loader(cfg, is_train=False)
    print("===" * 30)
    print("len train_loader==",len(train_loader),"    batch_size_train=",cfg.train.batch_size,"     all=",len(train_loader)*cfg.train.batch_size)
    print("len val_loader==", len(val_loader),"    batch_size_test=",cfg.test.batch_size)
    print("===" * 30)

    for epoch in range(begin_epoch, cfg.train.epoch):
        recorder.epoch = epoch
        trainer.train(epoch, train_loader, optimizer, recorder, cfg.iteration_save_my)
        scheduler.step()

        if (epoch + 1) % cfg.save_ep == 0:
            save_model(network, optimizer, scheduler, recorder, epoch, cfg.model_dir)

        if (epoch + 1) % cfg.eval_ep == 0:
            trainer.val(epoch, val_loader, evaluator, recorder)

    return network


def test(cfg, network):
    trainer = make_trainer(cfg, network)
    val_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)
    epoch = load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    trainer.val(epoch, val_loader, evaluator)


def main():
    network = make_network(cfg)
    print("===" * 25)
    print(cfg)
    print("===" * 25)
    if args.test:
        test(cfg, network)
    else:
        train(cfg, network)


if __name__ == "__main__":
    main()
