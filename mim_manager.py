import argparse
from mim import install, uninstall, list_package, get_model_info, download

def handle_install(args):
    if args.source.startswith('git+') or args.source.endswith('.git'):
        install(args.source)
    else:
        install(f"{args.source}=={args.version}" if args.version else args.source)

def handle_uninstall(args):
    uninstall(args.package)

def handle_list(args):
    list_package(args.all)

def handle_search(args):
    conditions = args.condition.split(',') if args.condition else None
    search_results = get_model_info(args.package, 
                   local=not args.remote,
                   models=args.model.split(',') if args.model else None,
                   training_datasets=args.dataset.split(',') if args.dataset else None,
                   filter_conditions=conditions,
                   sorted_fields=args.sort.split(',') if args.sort else None,
                   shown_fields=args.field.split(',') if args.field else None)
    
    print(search_results)
    print(search_results.index.values.tolist())
    print(search_results.columns.values.tolist())
    print(search_results['weight'].values.tolist())    


def handle_download(args):
    download(args.package, [args.config], dest_root=args.dest)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified MIM script")
    subparsers = parser.add_subparsers()

    # Install command
    parser_install = subparsers.add_parser('install')
    parser_install.add_argument('source', type=str)
    parser_install.add_argument('--version', type=str, default=None)
    parser_install.set_defaults(func=handle_install)

    # Uninstall command
    parser_uninstall = subparsers.add_parser('uninstall')
    parser_uninstall.add_argument('package', type=str)
    parser_uninstall.set_defaults(func=handle_uninstall)

    # List command
    parser_list = subparsers.add_parser('list')
    parser_list.add_argument('--all', action='store_true')
    parser_list.set_defaults(func=handle_list)

    # Search command
    parser_search = subparsers.add_parser('search')
    parser_search.add_argument('package', type=str)
    parser_search.add_argument('--remote', action='store_true')
    parser_search.add_argument('--model', type=str, default=None)
    parser_search.add_argument('--dataset', type=str, default=None)
    parser_search.add_argument('--condition', type=str, default=None)
    parser_search.add_argument('--sort', type=str, default=None)
    parser_search.add_argument('--field', type=str, default=None)
    parser_search.set_defaults(func=handle_search)

    # Download command
    parser_download = subparsers.add_parser('download')
    parser_download.add_argument('package', type=str)
    parser_download.add_argument('--config', type=str, required=True)
    parser_download.add_argument('--dest', type=str, default='.')
    parser_download.set_defaults(func=handle_download)

    args = parser.parse_args()
    args.func(args)

